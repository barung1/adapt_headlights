import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------- Dataset Loader ----------------------------
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

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_files))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        img = F.to_tensor(img)

        target = {"boxes": boxes, "labels": labels}

        return img, target

    def __len__(self):
        return len(self.image_files)

# ---------------------------- Data Loader ----------------------------
def get_dataloader(image_dir, annotation_dir, batch_size=4):
    dataset = VehicleDataset(image_dir, annotation_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# ---------------------------- Load Model ----------------------------
def get_model(num_classes=4):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------------------------- Training ----------------------------
def train_model(model, train_loader, device, num_epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)
    loss_list = []

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

        avg_loss = total_loss / len(train_loader)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "faster_rcnn.pth")
    print("✅ Model Trained & Saved")

    plt.plot(range(1, num_epochs+1), loss_list, marker='o')
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# ---------------------------- Evaluation ----------------------------
from torchvision.ops import box_iou

def evaluate_model(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    all_preds = []
    all_gts = []
    all_precisions = []
    all_recalls = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"]
                pred_labels = output["labels"]
                pred_scores = output["scores"]
                true_boxes = target["boxes"]
                true_labels = target["labels"]

                if len(pred_boxes) == 0 or len(true_boxes) == 0:
                    continue

                ious = box_iou(pred_boxes, true_boxes)
                matches = (ious > iou_threshold).sum().item()

                precision = matches / len(pred_boxes)
                recall = matches / len(true_boxes)
                all_precisions.append(precision)
                all_recalls.append(recall)

                all_preds.extend(pred_labels.cpu().numpy())
                all_gts.extend(true_labels.cpu().numpy())

    mean_precision = sum(all_precisions) / len(all_precisions)
    mean_recall = sum(all_recalls) / len(all_recalls)
    print(f"📊 Mean Precision: {mean_precision:.3f}")
    print(f"📊 Mean Recall: {mean_recall:.3f}")

    cm = confusion_matrix(all_gts, all_preds, labels=[1, 2, 3])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["TwoW", "FourW", "Headlight"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# ---------------------------- Sample Detections ----------------------------
def plot_detection_examples(model, dataset, device, num_images=6):
    model.eval()
    fig, axs = plt.subplots(2, num_images//2, figsize=(15, 6))

    for i in range(num_images):
        image, _ = dataset[i]
        image = image.to(device)
        with torch.no_grad():
            output = model([image])[0]

        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype('uint8')
        for box in output["boxes"]:
            x1, y1, x2, y2 = map(int, box.tolist())
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        axs[i // (num_images//2), i % (num_images//2)].imshow(img_np)
        axs[i // (num_images//2), i % (num_images//2)].axis('off')
    plt.tight_layout()
    plt.show()

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_dataloader("dataset/split/images/train", "dataset/split/annotations/train")
    val_loader = get_dataloader("dataset/split/images/val", "dataset/split/annotations/val")

    model = get_model()
    train_model(model, train_loader, device, num_epochs=10)

    model.load_state_dict(torch.load("faster_rcnn.pth"))
    evaluate_model(model, val_loader, device)

    val_dataset = VehicleDataset("dataset/split/images/val", "dataset/split/annotations/val")
    plot_detection_examples(model, val_dataset, device)
