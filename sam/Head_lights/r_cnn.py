import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torch.optim as optim
import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt


# **Dataset Loader**
class VehicleDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace(".jpg", ".xml"))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Parse XML annotation
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label == "two wheeler":
                labels.append(1)
            elif label == "four wheeler":
                labels.append(2)
            elif label == "headlight":
                labels.append(3)

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        # **Fix: If no objects found, return next image**
        if len(boxes) == 0:
            print(f"⚠️ Warning: No objects found in {img_path}, skipping...")
            return self.__getitem__((idx + 1) % len(self.image_files))

        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img = F.to_tensor(img)

        target = {"boxes": boxes, "labels": labels}

        return img, target

    def __len__(self):
        return len(self.image_files)



# **Function to Load Data**
def get_dataloader(image_dir, annotation_dir, batch_size=4):
    dataset = VehicleDataset(image_dir, annotation_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


# **Load Faster R-CNN Model**
def get_model(num_classes=4):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# **Train Model**
def train_model(model, train_loader, device, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)

    print("🚀 Training Started!")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "faster_rcnn.pth")
    print("✅ Model Training Completed & Saved!")


# **Inference & Visualization**
def run_inference(model, device, image_path):
    model.to(device)
    model.eval()

    # Load test image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Draw bounding boxes
    for box, label in zip(predictions[0]["boxes"], predictions[0]["labels"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# **Main Execution**
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader = get_dataloader("dataset/split/images/train", "dataset/split/annotations/train")
    print("✅ Dataset Loaded Successfully!")

    # Load model
    model = get_model()

    # Train model
    train_model(model, train_loader, device)

    # Load trained model
    model.load_state_dict(torch.load("faster_rcnn.pth"))

    # Test inference
    test_image_path = "dataset/split/images/test/frame_8335.jpg"
    run_inference(model, device, test_image_path)
