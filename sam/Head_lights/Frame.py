import cv2
import os

video_path = "My Movie.mp4"
output_folder = "Ext_Frames"
frame_rate = 5  # Extract every 5th frame

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_rate == 0:
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved: {frame_path}")  # Progress message

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Frames extracted successfully!")
