# 🍎 Fruit Object Detector using CNN (YOLOv8 Format)

A lightweight object detection system that detects fruits in images using a custom-trained Convolutional Neural Network (CNN). The model is trained on YOLOv8-style annotated fruit images.

---

## 📦 Features

- Detects 6 fruits: Apple, Banana, Grapes, Orange, Pineapple, Watermelon
- Trained on custom YOLO-format dataset (8479 images)
- Modular structure for flexibility and extension
- Real-time webcam inference support
- Model evaluation with F1-score, precision, recall, and confusion matrix

---

## 🗂️ Project Structure
object-detector/
├── data/
│ ├── train/ # Training images and labels
│ ├── valid/ # Validation images and labels
│ └── test/ # Test images and labels
├── dataset.py # Custom PyTorch Dataset class
├── model.py # CNN model for classification and bounding box prediction
├── train.py # Model training script
├── test_model.py # Inference on single image
├── evaluate.py # Performance metrics on test set
├── realtime_detect.py # Real-time webcam object detection
├── utils.py # Helper functions (IoU, drawing boxes, etc.)
└── README.md # Project documentation

## 📁 Dataset Format

- **YOLOv8 style** annotations: <class_id> <x_center> <y_center> <width> <height>
- Each `.txt` file has same name as corresponding `.jpg` image.
- Folder structure:
data/
├── train/
│ ├── images/
│ └── labels/
├── valid/
│ ├── images/
│ └── labels/
└── test/
├── images/
└── labels/
