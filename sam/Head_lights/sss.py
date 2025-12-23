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
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score


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
    all_scores = []
    all_classes = []
    label_map = {1: "two wheeler", 2: "four wheeler", 3: "headlight"}
    labels = list(label_map.keys())

    # For per-class analysis
    true_per_class = {label: [] for label in labels}
    pred_per_class = {label: [] for label in labels}
    scores_per_class = {label: [] for label in labels}

    # For counting detections
    class_counts = {label_map[label]: 0 for label in labels}

    total_images = 0
    detection_times = []

    for images, targets in dataloader:
        total_images += len(images)
        images = [img.to(device) for img in images]

        for images, targets in dataloader:
            total_images += len(images)
            images = [img.to(device) for img in images]

            if device.type == 'cuda':
                # CUDA timing
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.no_grad():
                    outputs = model(images)
                end_event.record()
                torch.cuda.synchronize()
                batch_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            else:
                # CPU timing
                import time
                start_time = time.time()
                with torch.no_grad():
                    outputs = model(images)
                batch_time = time.time() - start_time

            detection_times.append(batch_time / len(images))

        for output, target in zip(outputs, targets):
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            true_labels = target["labels"].cpu().numpy()

            # Store all prediction scores for ROC and PR curves
            for label, score in zip(pred_labels, pred_scores):
                all_scores.append(score)
                all_classes.append(label)
                scores_per_class[label].append(score)

            # Count detections by class
            valid_preds = pred_labels[pred_scores > score_threshold]
            for label in valid_preds:
                class_counts[label_map[label]] += 1

            # Filter by threshold for confusion matrix
            pred_labels = pred_labels[pred_scores > score_threshold]

            # For overall metrics
            y_pred.extend(pred_labels)
            y_true.extend(true_labels)

            # For per-class analysis
            for label in labels:
                true_positives = np.isin(label, true_labels).astype(int)
                pred_positives = np.isin(label, pred_labels).astype(int)
                true_per_class[label].extend([1] * sum(true_labels == label))
                pred_per_class[label].extend([1 if l == label else 0 for l in pred_labels])

    # 1. Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=[label_map[l] for l in labels])
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig_cm)

    # 2. Performance Metrics
    st.subheader("Performance Metrics")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Create a metrics dataframe
    metrics = {
        "Metric": ["Precision", "Recall", "F1 Score"],
        "Value": [precision, recall, f1]
    }
    metrics_df = pd.DataFrame(metrics)

    # Display metrics as a bar chart
    fig_metrics, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Metric", y="Value", data=metrics_df, ax=ax)
    ax.set_ylim(0, 1)
    plt.title("Overall Model Performance")
    st.pyplot(fig_metrics)

    # 3. Per-Class Metrics
    st.subheader("Per-Class Performance")
    class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_metrics = pd.DataFrame({
        "Class": [label_map[l] for l in labels],
        "Precision": class_precision,
        "Recall": class_recall,
        "F1 Score": class_f1
    })

    # Display table
    st.dataframe(per_class_metrics)

    # Display as grouped bar chart
    fig_class, ax = plt.subplots(figsize=(10, 6))
    per_class_metrics_melt = pd.melt(per_class_metrics, id_vars=['Class'],
                                     value_vars=['Precision', 'Recall', 'F1 Score'])
    sns.barplot(x="Class", y="value", hue="variable", data=per_class_metrics_melt, ax=ax)
    ax.set_ylim(0, 1)
    plt.title("Performance Metrics by Class")
    plt.legend(title="Metric")
    st.pyplot(fig_class)

    # 4. Detection Count by Class
    st.subheader("Detections by Class")
    fig_count, ax = plt.subplots(figsize=(8, 5))
    counts_df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    sns.barplot(x="Class", y="Count", data=counts_df, ax=ax)
    plt.title("Number of Detections by Class")
    st.pyplot(fig_count)

    # 5. Score Distribution
    st.subheader("Confidence Score Distribution")
    fig_dist, ax = plt.subplots(figsize=(10, 6))

    for label in labels:
        if len(scores_per_class[label]) > 0:  # Check if we have scores for this class
            sns.kdeplot(scores_per_class[label], label=label_map[label], fill=True, alpha=0.3)

    plt.axvline(x=score_threshold, color='r', linestyle='--', label=f'Threshold ({score_threshold})')
    plt.xlabel("Confidence Score")
    plt.ylabel("Density")
    plt.title("Distribution of Confidence Scores by Class")
    plt.legend()
    st.pyplot(fig_dist)

    # 6. Inference Speed
    st.subheader("Inference Speed")
    avg_time = np.mean(detection_times)
    st.write(f"**Average detection time:** {avg_time:.4f} seconds per image")
    st.write(f"**Frames per second:** {1 / avg_time:.2f} FPS")

    # Display raw metrics
    st.subheader("Raw Performance Numbers")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**Total images evaluated:** {total_images}")


# App starts here
model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

st.title("🚗 Vehicle Detection & Evaluation Dashboard")

mode = st.sidebar.selectbox("Choose mode", ["Single Image Inference", "Evaluate Test Set", "Advanced Analytics"])

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
            cv2.putText(img_np, f"{class_name} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        2)

        st.image(img_np, caption="Detected Image", channels="RGB")
        df = pd.DataFrame(results, columns=["Class", "Score", "Box"])
        st.dataframe(df)

        # Visual analysis of detections
        if results:
            st.subheader("Detection Analysis")
            class_counts = {}
            score_by_class = {}

            for res in results:
                cls = res[0]
                score = float(res[1])

                if cls not in class_counts:
                    class_counts[cls] = 0
                    score_by_class[cls] = []

                class_counts[cls] += 1
                score_by_class[cls].append(score)

            # Detection count
            fig1, ax1 = plt.subplots()
            ax1.bar(class_counts.keys(), class_counts.values())
            ax1.set_ylabel('Count')
            ax1.set_title('Detections by Class')
            st.pyplot(fig1)

            # Score distribution
            fig2, ax2 = plt.subplots()
            for cls, scores in score_by_class.items():
                ax2.boxplot(scores, labels=[cls])
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Confidence Score Distribution')
            st.pyplot(fig2)

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

elif mode == "Advanced Analytics":
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
            img_width = int(root.find("size").find("width").text)
            img_height = int(root.find("size").find("height").text)

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

            target = {"boxes": boxes, "labels": labels, "filename": self.image_files[idx],
                      "width": img_width, "height": img_height}
            return img, target

        def __len__(self):
            return len(self.image_files)


    def get_dataloader(image_dir, annotation_dir, batch_size=2):
        dataset = VehicleDataset(image_dir, annotation_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


    # Dataset path setup
    test_image_dir = "dataset/split/images/test"
    test_annot_dir = "dataset/split/annotations/test"

    st.subheader("Advanced Model Analytics")

    analytics_type = st.selectbox(
        "Choose Analysis Type",
        ["Error Analysis", "Box Size Distribution", "Threshold Impact"]
    )

    test_loader = get_dataloader(test_image_dir, test_annot_dir, batch_size=1)

    if analytics_type == "Error Analysis":
        st.write("Analyzing the most common error patterns in predictions...")

        # Process data and find error patterns
        model.eval()
        model.to(device)

        errors = []

        for images, targets in test_loader:
            images = [img.to(device) for img in images]

            with torch.no_grad():
                outputs = model(images)

            for output, target in zip(outputs, targets):
                filename = target["filename"]
                true_labels = target["labels"].cpu().numpy()
                true_boxes = target["boxes"].cpu().numpy()

                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                pred_boxes = output["boxes"].cpu().numpy()

                # Only consider predictions with scores above threshold
                valid_indices = pred_scores > score_thresh
                pred_labels = pred_labels[valid_indices]
                pred_boxes = pred_boxes[valid_indices]
                pred_scores = pred_scores[valid_indices]

                # Check for errors
                for true_label, true_box in zip(true_labels, true_boxes):
                    found_match = False
                    for pred_label, pred_box, pred_score in zip(pred_labels, pred_boxes, pred_scores):
                        # Calculate IoU
                        xA = max(true_box[0], pred_box[0])
                        yA = max(true_box[1], pred_box[1])
                        xB = min(true_box[2], pred_box[2])
                        yB = min(true_box[3], pred_box[3])

                        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                        true_area = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
                        pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)

                        iou = inter_area / float(true_area + pred_area - inter_area)

                        if iou > 0.5:  # Match found
                            found_match = True
                            if pred_label != true_label:
                                # Misclassification
                                errors.append({
                                    "filename": filename,
                                    "error_type": "misclassification",
                                    "true_label": int(true_label),
                                    "pred_label": int(pred_label),
                                    "confidence": float(pred_score),
                                    "iou": float(iou)
                                })
                            break

                    if not found_match:
                        # Miss (false negative)
                        errors.append({
                            "filename": filename,
                            "error_type": "miss",
                            "true_label": int(true_label),
                            "pred_label": None,
                            "confidence": None,
                            "iou": None
                        })

                # Check for false positives
                for pred_label, pred_box, pred_score in zip(pred_labels, pred_boxes, pred_scores):
                    found_match = False
                    for true_label, true_box in zip(true_labels, true_boxes):
                        # Calculate IoU
                        xA = max(true_box[0], pred_box[0])
                        yA = max(true_box[1], pred_box[1])
                        xB = min(true_box[2], pred_box[2])
                        yB = min(true_box[3], pred_box[3])

                        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                        true_area = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
                        pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)

                        iou = inter_area / float(true_area + pred_area - inter_area)

                        if iou > 0.5:  # Match found
                            found_match = True
                            break

                    if not found_match:
                        # False positive
                        errors.append({
                            "filename": filename,
                            "error_type": "false_positive",
                            "true_label": None,
                            "pred_label": int(pred_label),
                            "confidence": float(pred_score),
                            "iou": None
                        })

        # Convert to DataFrame
        errors_df = pd.DataFrame(errors)

        # Error type distribution
        st.subheader("Error Type Distribution")
        error_counts = errors_df["error_type"].value_counts().reset_index()
        error_counts.columns = ["Error Type", "Count"]

        fig_errors, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Error Type", y="Count", data=error_counts, ax=ax)
        plt.title("Distribution of Error Types")
        st.pyplot(fig_errors)

        # Class-wise error analysis for misclassifications
        misclass_df = errors_df[errors_df["error_type"] == "misclassification"]

        if not misclass_df.empty:
            st.subheader("Misclassification Analysis")

            # Create confusion matrix for misclassifications
            label_map = {1: "two wheeler", 2: "four wheeler", 3: "headlight"}

            true_labels = misclass_df["true_label"].tolist()
            pred_labels = misclass_df["pred_label"].tolist()

            if true_labels and pred_labels:
                unique_labels = sorted(list(set(true_labels + pred_labels)))
                cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
                disp = ConfusionMatrixDisplay(cm, display_labels=[label_map.get(l, "unknown") for l in unique_labels])

                fig_misclass, ax = plt.subplots(figsize=(8, 6))
                disp.plot(cmap="Reds", values_format="d", ax=ax)
                plt.title("Misclassification Matrix")
                st.pyplot(fig_misclass)

        # Show examples of errors if available
        st.subheader("Examples of Errors")
        st.write("Top errors by confidence score:")

        # Sort errors by confidence for false positives and misclassifications
        error_examples = errors_df[errors_df["confidence"].notna()].sort_values("confidence", ascending=False).head(5)

        if not error_examples.empty:
            st.dataframe(error_examples)
        else:
            st.write("No high-confidence errors found.")

    elif analytics_type == "Box Size Distribution":
        st.write("Analyzing the distribution of object sizes in the dataset...")

        model.eval()
        model.to(device)

        box_areas = []
        label_map = {1: "two wheeler", 2: "four wheeler", 3: "headlight"}

        for _, targets in test_loader:
            for target in targets:
                boxes = target["boxes"].cpu().numpy()
                labels = target["labels"].cpu().numpy()
                img_width = target["width"]
                img_height = target["height"]

                for box, label in zip(boxes, labels):
                    # Calculate relative box area (as percentage of image)
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area_pct = (width * height) / (img_width * img_height) * 100

                    box_areas.append({
                        "class": label_map[label],
                        "area_pct": area_pct,
                        "width": width,
                        "height": height,
                        "aspect_ratio": width / height if height > 0 else 0
                    })

        box_df = pd.DataFrame(box_areas)

        # Size distribution by class
        st.subheader("Object Size Distribution by Class")
        fig_size, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="class", y="area_pct", data=box_df, ax=ax)
        plt.title("Distribution of Object Sizes (% of Image Area)")
        plt.ylabel("Percentage of Image Area")
        st.pyplot(fig_size)

        # Aspect ratio distribution
        st.subheader("Object Aspect Ratio by Class")
        fig_aspect, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x="class", y="aspect_ratio", data=box_df, ax=ax)
        plt.title("Distribution of Object Aspect Ratios (width/height)")
        plt.ylabel("Aspect Ratio (width/height)")
        st.pyplot(fig_aspect)

        # 2D plot of width vs height
        st.subheader("Width vs Height by Class")
        fig_wh, ax = plt.subplots(figsize=(10, 8))

        for cls in box_df["class"].unique():
            cls_data = box_df[box_df["class"] == cls]
            ax.scatter(cls_data["width"], cls_data["height"], alpha=0.7, label=cls)

        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.title("Object Dimensions by Class")
        plt.legend()
        st.pyplot(fig_wh)

    elif analytics_type == "Threshold Impact":
        st.write("Analyzing the impact of confidence threshold on model performance...")

        # Evaluate model at different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        metrics = []

        model.eval()
        model.to(device)

        # Collect all predictions and ground truths
        all_preds = []
        all_targets = []

        for images, targets in test_loader:
            images = [img.to(device) for img in images]

            with torch.no_grad():
                outputs = model(images)

            all_preds.extend(outputs)
            all_targets.extend(targets)

        # Calculate metrics at different thresholds
        for threshold in thresholds:
            y_true = []
            y_pred = []

            for output, target in zip(all_preds, all_targets):
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()
                true_labels = target["labels"].cpu().numpy()

                # Filter by threshold
                pred_labels = pred_labels[pred_scores > threshold]

                y_pred.extend(pred_labels)
                y_true.extend(true_labels)

            if len(y_true) > 0 and len(y_pred) > 0:
                precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
                recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
                f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

                metrics.append({
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })

        metrics_df = pd.DataFrame(metrics)

        # Plot precision-recall trade-off
        st.subheader("Precision-Recall Trade-off")
        fig_pr, ax = plt.subplots(figsize=(10, 6))

        ax.plot(metrics_df["threshold"], metrics_df["precision"], marker='o', label="Precision")
        ax.plot(metrics_df["threshold"], metrics_df["recall"], marker='s', label="Recall")
        ax.plot(metrics_df["threshold"], metrics_df["f1"], marker='^', label="F1 Score")

        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Impact of Confidence Threshold on Model Performance")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add vertical line at current threshold
        ax.axvline(x=score_thresh, color='r', linestyle='--', label=f'Current ({score_thresh})')
        ax.legend()

        st.pyplot(fig_pr)

        # Recommend optimal threshold
        f1_max_idx = metrics_df["f1"].idxmax()
        optimal_threshold = metrics_df.loc[f1_max_idx, "threshold"]

        st.write(f"**Optimal threshold for F1 score:** {optimal_threshold:.2f}")
        st.write(f"**Precision at optimal threshold:** {metrics_df.loc[f1_max_idx, 'precision']:.4f}")
        st.write(f"**Recall at optimal threshold:** {metrics_df.loc[f1_max_idx, 'recall']:.4f}")
        st.write(f"**F1 Score at optimal threshold:** {metrics_df.loc[f1_max_idx, 'f1']:.4f}")

        # Show metrics table
        st.subheader("Detailed Metrics by Threshold")
        st.dataframe(metrics_df.style.highlight_max(subset=["precision", "recall", "f1"]))

        # Number of detections vs threshold
        st.subheader("Detection Count vs Threshold")

        detection_counts = []
        for threshold in thresholds:
            count = 0
        for output in all_preds:
            pred_scores = output["scores"].cpu().numpy()
        count += sum(pred_scores > threshold)

        detection_counts.append({
            "threshold": threshold,
            "detection_count": count
        })

        counts_df = pd.DataFrame(detection_counts)

        fig_counts, ax = plt.subplots(figsize=(10, 6))
        ax.plot(counts_df["threshold"], counts_df["detection_count"], marker='o')
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Number of Detections")
        ax.set_title("Impact of Threshold on Detection Count")
        ax.grid(True, alpha=0.3)

        # Add vertical line at current threshold
        ax.axvline(x=score_thresh, color='r', linestyle='--', label=f'Current ({score_thresh})')
        ax.legend()

        st.pyplot(fig_counts)