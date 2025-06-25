import cv2
import torch
from torchvision import transforms
from model import FruitDetector  # Make sure your model file uses this name
from PIL import Image
import numpy as np

# Define your classes
CLASS_NAMES = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']
NUM_CLASSES = len(CLASS_NAMES)

# Initialize the model with num_classes
model = FruitDetector(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("fruit_detector.pth", map_location=torch.device('cpu')))
model.eval()

# Define transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Add normalization if used in training
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert to PIL format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Apply transform
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        bbox_pred, class_logits = model(input_tensor)
        probs = torch.softmax(class_logits, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_idx].item()
        bbox = bbox_pred[0].numpy()

    if confidence > 0.7:
        h, w, _ = frame.shape
        cx, cy, bw, bh = bbox
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        label = f"{CLASS_NAMES[class_idx]} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Fruit Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
