from dataset import YOLODataset
import matplotlib.pyplot as plt
import cv2
import os
import random

# Path to your data
image_dir = "data/train/images"
label_dir = "data/train/labels"

# Class names as per your dataset
class_names = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']

# Load the dataset
dataset = YOLODataset(image_dir, label_dir)

# Get a random sample from dataset
index = random.randint(0, len(dataset) - 1)
image, boxes = dataset[index]

# Convert image tensor to numpy for OpenCV/Matplotlib
image_np = image.permute(1, 2, 0).numpy().copy()
h, w, _ = image_np.shape

# Draw bounding boxes on the image
for box in boxes:
    class_id, x_center, y_center, bw, bh = box.tolist()

    # Convert normalized to absolute pixel values
    x_center *= w
    y_center *= h
    bw *= w
    bh *= h

    x1 = int(x_center - bw / 2)
    y1 = int(y_center - bh / 2)
    x2 = int(x_center + bw / 2)
    y2 = int(y_center + bh / 2)

    # Draw rectangle + label
    label = class_names[int(class_id)]
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Show image using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(image_np)
plt.axis('off')
plt.title(f"Random Sample #{index}")
plt.show()
