# app.py
import streamlit as st
import torch
import torchvision
import numpy as np
import cv2
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score

# Load the model
@st.cache_resource
def load_model():
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
    model.load_state_dict(torch.load("faster_rcnn.pth", map_location="cpu"))
    model.eval()
    return model

# Evaluation function
def evaluate_model(model, dataloader, device, score_threshold=0.5):
    model.eval()
    model.to(device)

    y_true, y_pred = [], []
    label_map = {1: "two wheeler", 2: "four wheeler", 3: "headlight"}
    labels = list(label_map.keys())

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, targets):
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            pred_labels = pred_labels[pred_scores > score_threshold]
            true_labels = target["labels"].cpu().numpy()
            y_pred.extend(pred_labels)
            y_true.extend(true_labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=[label_map[l] for l in labels])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    st.pyplot(plt.gcf())

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    st.write(f"**Precision:** {precision:.2f}")
    st.write(f"**Recall:** {recall:.2f}")
    st.write(f"**F1 Score:** {f1:.2f}")

# App starts here
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("🚗 Vehicle Detection & Evaluation Dashboard")

mode = st.sidebar.selectbox("Choose mode", ["Single Image Inference", "Evaluate Test Set"])

score_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

if mode == "Single Image Inference":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_tensor = F.to_tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)[0]

        label_map = {1: "two wheeler", 2: "four wheeler", 3: "headlight"}
        results = []

        for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
            if score < score_thresh:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = label_map.get(label.item(), "unknown")
            results.append([class_name, f"{score:.2f}", f"({x1}, {y1}, {x2}, {y2})"])
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        st.image(img_np, caption="Detected Image", channels="RGB")
        df = pd.DataFrame(results, columns=["Class", "Score", "Box"])
        st.dataframe(df)

elif mode == "Evaluate Test Set":
    import os
    from torch.utils.data import Dataset, DataLoader
    import xml.etree.ElementTree as ET

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

    def get_dataloader(image_dir, annotation_dir, batch_size=2):
        dataset = VehicleDataset(image_dir, annotation_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Dataset path setup
    test_image_dir = "dataset/split/images/test"
    test_annot_dir = "dataset/split/annotations/test"
    test_loader = get_dataloader(test_image_dir, test_annot_dir)

    st.subheader("📊 Evaluation on Test Set")
    evaluate_model(model, test_loader, device, score_threshold=score_thresh)
