from dataset import YOLODataset
from utils import draw_boxes, calculate_iou
import matplotlib.pyplot as plt
import random

# Path to data
image_dir = "data/train/images"
label_dir = "data/train/labels"
class_names = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']

# Load dataset
dataset = YOLODataset(image_dir, label_dir)

# Pick a random image
index = random.randint(0, len(dataset) - 1)
image, boxes = dataset[index]

# Test draw_boxes()
image_with_boxes = draw_boxes(image, boxes, class_names)

# Display image
plt.figure(figsize=(8, 8))
plt.imshow(image_with_boxes)
plt.axis('off')
plt.title(f"Image #{index} with Bounding Boxes")
plt.show()

# Test calculate_iou()
if len(boxes) >= 2:
    iou = calculate_iou(boxes[0][1:], boxes[1][1:])  # Skip class_id
    print(f"IoU between first two boxes: {iou:.4f}")
else:
    print("Only one object in image — skipping IoU test.")
