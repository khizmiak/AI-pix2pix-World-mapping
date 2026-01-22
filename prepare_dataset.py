import os
from PIL import Image
import numpy as np

def save_paired_images(satellite_path, map_path, output_dir, filename):
    """
    Concatenate satellite and map images horizontally and save as a single image.
    Assumes images are RGB and same height.
    """
    satellite = Image.open(satellite_path).convert('RGB')
    map_img = Image.open(map_path).convert('RGB')

    # Resize to 256x256 if needed, but keep aspect for concat
    # For Pix2Pix, typically 256x256, but dataset.py resizes to 256x256 anyway
    # Concatenate horizontally: satellite | map
    combined = Image.new('RGB', (satellite.width + map_img.width, satellite.height))
    combined.paste(satellite, (0, 0))
    combined.paste(map_img, (satellite.width, 0))

    os.makedirs(output_dir, exist_ok=True)
    combined.save(os.path.join(output_dir, filename))

# Example usage:
# save_paired_images('satellite.jpg', 'map.jpg', 'datasets', 'pair1.jpg')

# If you have directories of unpaired images, you can loop and pair them manually.
# For example, if satellite_dir and map_dir have matching filenames:
# satellite_dir = 'path/to/satellites'
# map_dir = 'path/to/maps'
# output_dir = 'datasets'
# for file in os.listdir(satellite_dir):
#     if file in os.listdir(map_dir):
#         save_paired_images(os.path.join(satellite_dir, file), os.path.join(map_dir, file), output_dir, file)
