import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import config


class MapDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = [f for f in os.listdir(self.root) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = os.path.join(self.root, img_file)
        image = Image.open(img_path).convert("RGB")

        w, h = image.size
        half = w // 2
        input_pil = image.crop((0, 0, half, h))
        target_pil = image.crop((half, 0, w, h))

        if config.USE_ALBUMENTATIONS:
            input_image = np.array(input_pil)
            target_image = np.array(target_pil)
            aug = config.both_transform(image=input_image, image0=target_image)
            input_image, target_image = aug["image"], aug["image0"]
            input_image = config.transform_input(image=input_image)["image"]
            target_image = config.transform_output(image=target_image)["image"]
            return input_image, target_image

        return config.transform_torchvision(input_pil, target_pil)
