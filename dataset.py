import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class YOLODataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # === Load image ===
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(self.label_dir, image_filename.replace(".jpg", ".txt"))

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        # === Load labels ===
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    boxes.append([class_id, x_center, y_center, width, height])

        # Convert to tensor (can be empty if no labels)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

        return image_tensor, boxes_tensor
