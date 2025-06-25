import os
from collections import Counter

# Define your class labels (IDs must match your YOLO dataset)
class_names = ['Apple', 'Grapes', 'Pineapple', 'Orange', 'Banana', 'Watermelon']

label_folder = "data/train/labels"  # Change if needed

class_counts = Counter()

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(label_folder, filename), "r") as file:
            for line in file:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

# Display counts by class name
print("\n🧾 Class Distribution in Training Set:\n")
for class_id, count in sorted(class_counts.items()):
    print(f"{class_names[class_id]} ({class_id}): {count} instances")
