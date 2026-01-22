import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MapDataset
import config
from utils import save_checkpoint, load_checkpoint, save_examples
from generator import Generator
from discriminator import Discriminator


def _train_step_cuda(discriminator, generator, x, y, l1, bce, disc_opt, gen_opt, g_scaler, d_scaler):
    with torch.cuda.amp.autocast():
        y_fake = generator(x)
        D_real = discriminator(x, y)
        D_fake = discriminator(x, y_fake.detach())
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss).mean()

    discriminator.zero_grad()
    d_scaler.scale(D_loss).backward()
    d_scaler.step(disc_opt)
    d_scaler.update()

    with torch.cuda.amp.autocast():
        D_fake = discriminator(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

    generator.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(gen_opt)
    g_scaler.update()
    return D_loss, G_loss, L1


def _train_step_cpu(discriminator, generator, x, y, l1, bce, disc_opt, gen_opt):
    y_fake = generator(x)
    D_real = discriminator(x, y)
    D_fake = discriminator(x, y_fake.detach())
    D_real_loss = bce(D_real, torch.ones_like(D_real))
    D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
    D_loss = (D_real_loss + D_fake_loss).mean()

    discriminator.zero_grad()
    D_loss.backward()
    disc_opt.step()

    D_fake = discriminator(x, y_fake)
    G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
    L1_loss = l1(y_fake, y) * config.L1_LAMBDA
    G_loss = G_fake_loss + L1_loss

    generator.zero_grad()
    G_loss.backward()
    gen_opt.step()
    return D_loss, G_loss, L1_loss


def train(discriminator, generator, loader, disc_opt, gen_opt, l1, bce, g_scaler, d_scaler, epoch, use_cuda):
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        if use_cuda:
            D_loss, G_loss, L1 = _train_step_cuda(
                discriminator, generator, x, y, l1, bce, disc_opt, gen_opt, g_scaler, d_scaler
            )
        else:
            D_loss, G_loss, L1 = _train_step_cpu(discriminator, generator, x, y, l1, bce, disc_opt, gen_opt)

        loop.set_postfix(
            D_loss=float(D_loss.item()),
            G_loss=float(G_loss.item()),
            L1_loss=float(L1.item())
        )
                


def main():
    discriminator = Discriminator(in_channels=3).to(config.DEVICE)
    generator = Generator(in_channels=3).to(config.DEVICE)
    
    disc_opt = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    gen_opt = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, generator, gen_opt, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, discriminator, disc_opt, config.LEARNING_RATE)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()
    else:
        g_scaler = d_scaler = None
        print("CUDA not available. Training on CPU (no AMP).")

    train_dataset = MapDataset(root="datasets/maps/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataset = MapDataset(root="datasets/maps/val")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train(discriminator, generator, train_loader, disc_opt, gen_opt, L1_LOSS, BCE, g_scaler, d_scaler, epoch, use_cuda)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(generator, gen_opt, filename=config.CHECKPOINT_GEN)
            save_checkpoint(discriminator, disc_opt, filename=config.CHECKPOINT_DISC)

        save_examples(generator, val_loader, epoch, folder="examples")
    


if __name__ =="__main__":
    main()