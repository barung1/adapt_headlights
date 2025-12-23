import torch
import cv2
import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load trained Faster R-CNN model
def load_model(model_path, num_classes=4):
    """
    Load the trained Faster R-CNN model.
    :param model_path: Path to the saved model weights (.pth file)
    :param num_classes: Number of classes in the dataset (including background)
    :return: Loaded model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # Modify the classifier to match number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    return model


# Function to run inference on an image
def run_inference(model, image_path, conf_threshold=0.5):
    """
    Runs inference on a single image and displays the results.
    :param model: Trained Faster R-CNN model
    :param image_path: Path to the image file
    :param conf_threshold: Confidence threshold for displaying predictions
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image '{image_path}' not found! Check file path.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Failed to read image '{image_path}'. It may be corrupted.")
        return

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Extract bounding boxes, labels, and scores
    pred_boxes = predictions[0]["boxes"].cpu().numpy()
    pred_labels = predictions[0]["labels"].cpu().numpy()
    pred_scores = predictions[0]["scores"].cpu().numpy()

    # Draw bounding boxes on the image
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < conf_threshold:
            continue  # Skip low-confidence detections

        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)  # Green box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"Class {label} ({score:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detections
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Detections for {os.path.basename(image_path)}")
    plt.show()


# Function to run inference on all images in a folder
def run_inference_on_folder(model, folder_path, conf_threshold=0.5):
    """
    Runs inference on all images in a folder.
    :param model: Trained Faster R-CNN model
    :param folder_path: Path to the test images folder
    :param conf_threshold: Confidence threshold for displaying predictions
    """
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found!")
        return

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"⚠️ No images found in '{folder_path}'!")
        return

    print(f"📷 Found {len(image_files)} images in '{folder_path}'. Running inference...")

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"🖼️ Processing {image_file}...")
        run_inference(model, image_path, conf_threshold)


# Main function
if __name__ == "__main__":
    model_path = "faster_rcnn.pth"  # Change this to the correct path of your model
    test_folder_path = "dataset/split/images/test"  # Change this to the path of your test images folder

    # Load trained model
    model = load_model(model_path)
    print("✅ Model Loaded Successfully!")

    # Run inference on all images in the test folder
    run_inference_on_folder(model, test_folder_path)
