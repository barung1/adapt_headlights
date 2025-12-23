import os
import shutil
import random

# Define paths
dataset_dir = "dataset/images"  # Path where your images are stored
annotations_dir = "dataset/annotations"  # Path where annotations are stored
output_dir = "dataset/split"  # Output directory for train/val/test

# Create train, val, test folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations", split), exist_ok=True)

# Get all image filenames
images = [f for f in os.listdir(dataset_dir) if f.endswith(".jpg")]

# Shuffle and split
random.shuffle(images)
train_split = int(0.7 * len(images))
val_split = int(0.9 * len(images))

train_files = images[:train_split]
val_files = images[train_split:val_split]
test_files = images[val_split:]

# Function to move files
def move_files(files, split):
    for file in files:
        shutil.move(os.path.join(dataset_dir, file), os.path.join(output_dir, "images", split, file))
        annotation_file = file.replace(".jpg", ".xml")  # Modify if using JSON or TXT annotations
        shutil.move(os.path.join(annotations_dir, annotation_file), os.path.join(output_dir, "annotations", split, annotation_file))

# Move files
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split successfully!")
