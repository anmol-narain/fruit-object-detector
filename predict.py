import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import os
from model import FruitDetector
from utils import draw_boxes
from PIL import Image
import random

# -------------------- CONFIG --------------------
MODEL_PATH = "fruit_detector.pth"
IMAGE_DIR = "data/test/images"  # your test images folder
LABELS = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# -------------------- LOAD MODEL --------------------
model = FruitDetector(num_classes=len(LABELS)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------- TRANSFORM --------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

# -------------------- LOAD RANDOM IMAGE --------------------
image_files = os.listdir(IMAGE_DIR)
random_image_name = random.choice(image_files)
image_path = os.path.join(IMAGE_DIR, random_image_name)

original_img = Image.open(image_path).convert("RGB")
input_tensor = transform(original_img).unsqueeze(0).to(DEVICE)

# -------------------- PREDICTION --------------------
with torch.no_grad():
    bbox_pred, class_logits = model(input_tensor)

# Convert bbox to numpy and rescale to original image size
bbox = bbox_pred[0].cpu().numpy()
bbox = bbox * IMG_SIZE  # since training was on 224x224

# Get predicted class
probabilities = torch.softmax(class_logits, dim=1)
class_idx = torch.argmax(probabilities, dim=1).item()
class_name = LABELS[class_idx]
conf = probabilities[0][class_idx].item()

# -------------------- DRAW --------------------
img_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_cv, (IMG_SIZE, IMG_SIZE))

x, y, w, h = bbox
x1 = int((x - w/2))
y1 = int((y - h/2))
x2 = int((x + w/2))
y2 = int((y + h/2))

cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img_resized, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# -------------------- SHOW --------------------
plt.imshow(img_resized)
plt.title("Prediction")
plt.axis("off")
plt.show()
