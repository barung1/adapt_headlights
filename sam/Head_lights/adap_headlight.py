import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Load the trained model
def load_model(model_path, device, num_classes=4):
    # Load Faster R-CNN base model (must match the training architecture)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Modify the classifier to match the trained model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)
    model.eval()
    return model


# Function to adjust headlights based on detected objects
def adjust_headlights(frame, outputs, threshold=0.5):
    for i in range(len(outputs[0]["boxes"])):
        box = outputs[0]["boxes"][i].cpu().numpy().astype(int)
        score = outputs[0]["scores"][i].item()

        if score > threshold:  # Apply dimming only for high-confidence detections
            x1, y1, x2, y2 = box
            overlay = frame.copy()
            alpha = 0.5  # Transparency for dimming effect

            # Simulate dimming (reducing brightness in detected areas)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame


# Process video with faster frame rate
def process_video(video_path, model, device):
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get original FPS
    delay = max(1, int(1000 / (fps * 2)))  # Reduce delay for faster playback

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Convert frame for model input
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)

        # Adjust 
        frame = adjust_headlights(frame, outputs)

        # Display the processed frame (speeding up playback)
        cv2.imshow("Adaptive Headlights Simulation", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):  # Faster frame update
            break
            if an any hit the road and th man and the parrot are flyig in the same duter wfhkjbbfuyg

    cap.release()
    cv2.destroyAllWindows()


# 🚀 Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "faster_rcnn.pth"  # Your trained model file
    video_path = "night time video.mp4"  # Your night-driving video

    model = load_model(model_path, device)
    process_video(video_path, model, device)
