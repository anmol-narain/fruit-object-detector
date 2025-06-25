import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import YOLODataset
from model import FruitDetector

# === CONFIGURATION ===
image_dir = "data/train/images"
label_dir = "data/train/labels"
class_names = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']

num_classes = len(class_names)
image_size = 224
batch_size = 8
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === TRANSFORM PIPELINE ===
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

# === CUSTOM COLLATE FUNCTION ===
# Prevents PyTorch from trying to stack variable-length labels
def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)  # target: [num_objects, 5]
    return torch.stack(images), targets

# === LOAD DATASET
full_dataset = YOLODataset(image_dir, label_dir, transform=transform)
# subset = Subset(full_dataset, range(2000))  # Faster training

# === USE COLLATE_FN HERE (IMPORTANT!)
train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# === MODEL SETUP ===
model = FruitDetector(num_classes=num_classes).to(device)
criterion_bbox = nn.MSELoss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === TRAINING LOOP ===
print("🚀 Training started...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)

        # Use only the first object per image
        bboxes = torch.stack([t[0, 1:] for t in targets]).to(device)      # shape: [B, 4]
        class_ids = torch.tensor([int(t[0, 0]) for t in targets]).to(device)  # shape: [B]

        # Forward pass
        pred_bbox, pred_class_logits = model(images)

        # Compute loss
        loss_bbox = criterion_bbox(pred_bbox, bboxes)
        loss_class = criterion_class(pred_class_logits, class_ids)
        loss = loss_bbox + loss_class

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Step {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch [{epoch+1}/{num_epochs}] Completed - Avg Loss: {avg_loss:.4f}")

# === SAVE MODEL ===
torch.save(model.state_dict(), "fruit_detector.pth")
print("💾 Model saved to fruit_detector.pth")
