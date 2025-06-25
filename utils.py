import cv2
import torch

def draw_boxes(image_tensor, boxes, class_names):
    """
    Draw bounding boxes (YOLO format) on an image tensor and return NumPy RGB image.
    
    Args:
        image_tensor (Tensor): shape (3, H, W), float in [0,1]
        boxes (Tensor): shape (N, 5), format = [class_id, x_center, y_center, width, height]
        class_names (list): list of class name strings

    Returns:
        image (ndarray): RGB image with boxes drawn, shape (H, W, 3)
    """
    # Convert image from torch (C, H, W) to NumPy (H, W, C)
    image = image_tensor.permute(1, 2, 0).numpy().copy()
    h, w, _ = image.shape

    for box in boxes:
        class_id, x_center, y_center, bw, bh = box.tolist()

        # Convert from normalized to pixel coordinates
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        # Convert center to corner coordinates
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        # Draw rectangle and label
        label = class_names[int(class_id)]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image

def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two YOLO-format boxes.

    Args:
        box1, box2 (list or Tensor): [x_center, y_center, width, height]

    Returns:
        iou (float): value between 0 and 1
    """
    def to_corners(box):
        x, y, w, h = box
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    x1_min, y1_min, x1_max, y1_max = to_corners(box1)
    x2_min, y2_min, x2_max, y2_max = to_corners(box2)

    # Calculate intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    # Return IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou
