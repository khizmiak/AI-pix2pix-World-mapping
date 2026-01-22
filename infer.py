import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import config
from generator import Generator
from utils import load_checkpoint
import torch.optim as optim
from torchvision.utils import save_image
import os

def inference(generator, image_path, output_path):
    """
    Generate a map from a satellite image.
    
    Args:
        generator: Trained generator model
        image_path: Path to input satellite image
        output_path: Path to save generated map
    """
    # Load and preprocess input image
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Apply same transforms as training
    augmentations = config.both_transform(image=image)
    input_image = augmentations["image"]
    input_image = config.transform_input(image=input_image)["image"]
    
    # Add batch dimension and move to device
    input_image = input_image.unsqueeze(0).to(config.DEVICE)
    
    # Generate map
    generator.eval()
    with torch.no_grad():
        generated_map = generator(input_image)
        # Denormalize: from [-1, 1] to [0, 1]
        generated_map = generated_map * 0.5 + 0.5
    
    # Save result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    save_image(generated_map, output_path)
    print(f"Generated map saved to: {output_path}")


def main():
    # Initialize generator
    generator = Generator(in_channels=3).to(config.DEVICE)
    gen_opt = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    
    # Load checkpoint
    if os.path.exists(config.CHECKPOINT_GEN):
        load_checkpoint(config.CHECKPOINT_GEN, generator, gen_opt, config.LEARNING_RATE)
        print(f"Loaded checkpoint from {config.CHECKPOINT_GEN}")
    else:
        print(f"Warning: Checkpoint {config.CHECKPOINT_GEN} not found. Using untrained model.")
    
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python infer.py <input_image_path> <output_image_path>")
        print("Example: python infer.py satellite.jpg generated_map.png")
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at {input_path}")
        return
    
    inference(generator, input_path, output_path)


if __name__ == "__main__":
    main()
