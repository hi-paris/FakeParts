# data/custom_dataset.py

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomRealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, opt):
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

        self.transform = transforms.Compose([
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, img_path  # <-- now returning image pat