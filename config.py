import torch
import random

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 4  # Use 16 for GPU training; 4 for CPU test
NUM_WORKERS = 0  # Use 2+ for faster loading when CUDA available; 0 avoids spawn issues on Windows
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 1  # Set to 500 for full training; 1 for quick test
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    USE_ALBUMENTATIONS = True
    both_transform = A.Compose(
        [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
    )
    transform_input = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3, max_pixel_value=255.0),
        ToTensorV2(),
    ])
    transform_output = A.Compose([
        A.Normalize(mean=[0.5] * 3, std=[0.5] * 3, max_pixel_value=255.0),
        ToTensorV2(),
    ])
except ImportError:
    USE_ALBUMENTATIONS = False
    from torchvision import transforms
    import torchvision.transforms.functional as TF

    _size = (256, 256)
    _norm = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    _jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def transform_torchvision(input_pil, target_pil):
        inp = input_pil.resize(_size)
        tgt = target_pil.resize(_size)
        if random.random() < 0.5:
            inp = TF.hflip(inp)
            tgt = TF.hflip(tgt)
        inp = _jitter(inp)
        inp = _norm(TF.to_tensor(inp))
        tgt = _norm(TF.to_tensor(tgt))
        return inp, tgt
