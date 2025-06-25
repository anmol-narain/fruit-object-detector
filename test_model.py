import torch
import random
import os
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from model import FruitDetector
from dataset import YOLODataset
from utils import draw_boxes

# === CONFIGURATION ===
image_dir = "data/train/images"  # Try training images first for sanity check
model_path = "fruit_detector.pth"
class_names = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']
image_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODEL ===
model = FruitDetector(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === RANDOM IMAGE FROM FOLDER ===
image_files = os.listdir(image_dir)
random_image_file = random.choice(image_files)
image_path = os.path.join(image_dir, random_image_file)

# === LOAD IMAGE ===
original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Save original shape for bbox rescaling
orig_h, orig_w, _ = original_image.shape

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])
input_tensor = transform(original_image_rgb).unsqueeze(0).to(device)

# === INFERENCE ===
with torch.no_grad():
    pred_bbox, pred_logits = model(input_tensor)
    pred_bbox = pred_bbox[0].cpu().numpy()       # [x_center, y_center, w, h]
    pred_probs = torch.softmax(pred_logits, dim=1)[0]
    pred_class = torch.argmax(pred_probs).item()
    pred_label = class_names[pred_class]
    confidence = pred_probs[pred_class].item()

# === DEBUG: PRINT RAW OUTPUT ===
print("🔍 Predicted Bounding Box:", pred_bbox)
print("🎯 Predicted Class:", pred_label)
print("🔥 Confidence:", round(confidence, 3))

# === RESCALE BBOX TO ORIGINAL IMAGE SIZE ===
x_center = pred_bbox[0] * orig_w
y_center = pred_bbox[1] * orig_h
bbox_width = pred_bbox[2] * orig_w
bbox_height = pred_bbox[3] * orig_h

x1 = int(x_center - bbox_width / 2)
y1 = int(y_center - bbox_height / 2)
x2 = int(x_center + bbox_width / 2)
y2 = int(y_center + bbox_height / 2)

# === DRAW PREDICTION ===
cv2.rectangle(original_image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(original_image_rgb, f"{pred_label} ({round(confidence*100)}%)",
            (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

# === DISPLAY RESULT ===
plt.imshow(original_image_rgb)
plt.title(f"{pred_label} ({round(confidence * 100)}%)")
plt.axis("off")
plt.show()
