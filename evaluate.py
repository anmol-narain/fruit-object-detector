import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from dataset import YOLODataset
from model import FruitDetector

# ✅ Step 1: Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ Step 2: Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Step 3: Define custom collate function
def custom_collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return torch.stack(images), targets  # keep targets as list of tensors

# ✅ Step 4: Load test dataset
test_dataset = YOLODataset('data/test/images', 'data/test/labels', transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

# ✅ Step 5: Load trained model
model = FruitDetector(num_classes=6).to(device)
model.load_state_dict(torch.load("fruit_detector.pth", map_location=device))
model.eval()

# ✅ Step 6: Class names (YOLO format: 0–5)
class_names = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']

# ✅ Step 7: Inference + Collect predictions
all_preds = []
all_targets = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        _, class_logits = model(images)  # ✅ Unpack only class predictions

        _, preds = torch.max(class_logits, 1)  # ✅ Take argmax of logits

        for t in targets:
            class_id = int(t[0][0].item())  # First object's class
            all_targets.append(class_id)

        all_preds.extend(preds.cpu().numpy())


# ✅ Step 8: Show metrics
print("\n📊 Classification Report:")
print(classification_report(all_targets, all_preds, target_names=class_names))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(all_targets, all_preds))