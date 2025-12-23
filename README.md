Adaptive Headlight System Using Faster R-CNN

This project implements a software-based adaptive headlight system that uses computer vision and deep learning to detect vehicles and headlight glare in nighttime driving scenarios. The system leverages Faster R-CNN to identify glare sources and evaluates performance using detection accuracy, luminance-based glare analysis, and inference time metrics.

The goal is to demonstrate how AI-driven perception can improve nighttime road safety without relying on specialized hardware sensors.

🚀 Key Features

Faster R-CNN–based detection of:

Two-wheelers

Four-wheelers

Headlight regions

Robust object detection in low-light conditions

Software-based adaptive headlight simulation using OpenCV

Quantitative evaluation using:

mAP, precision, recall

Average headlight luminance

Frame processing time

Graphical comparison with popular object detection models

🧠 Motivation

Glare from oncoming vehicle headlights is a major cause of reduced visibility and driver discomfort during nighttime driving. Traditional adaptive headlight systems rely on hardware sensors and fixed thresholds, which often fail in complex traffic environments. This project explores a computer vision–based approach that dynamically detects glare sources and simulates adaptive beam control using deep learning.

🏗️ System Architecture
Nighttime Image / Video Frame
            ↓
Faster R-CNN Object Detection
            ↓
Vehicle & Headlight Localization
            ↓
Luminance-Based Glare Analysis
            ↓
Adaptive Headlight Simulation
            ↓
Performance Evaluation & Visualization

⚙️ Methodology
Object Detection

Faster R-CNN with ResNet-50 FPN backbone

Pretrained on COCO and fine-tuned on a custom dataset

Pascal VOC–style XML annotations

Glare Analysis

Headlight regions extracted from model predictions

Average luminance computed using grayscale intensity

High-luminance regions treated as glare sources

Evaluation Metrics

Detection accuracy (mAP, precision, recall)

Average headlight luminance

Inference time per frame

📂 Dataset Structure
dataset/
 └── split/
     ├── images/
     │   ├── train/
     │   └── test/
     └── annotations/
         ├── train/
         └── test/


Class Labels

1 → Two-wheeler

2 → Four-wheeler

3 → Headlight

🛠 Technologies Used

Python

PyTorch & Torchvision

OpenCV

TorchMetrics

Matplotlib

⚙️ Installation
pip install torch torchvision torchmetrics opencv-python matplotlib

▶️ How to Run
1️⃣ Train the Model
python main.py


This will:

Load the training dataset

Train Faster R-CNN for 10 epochs

Save the trained model as faster_rcnn.pth

2️⃣ Evaluate the Model

During execution, the script:

Computes detection metrics (mAP, precision, recall)

Measures average headlight luminance

Calculates inference time

Generates comparison graphs

3️⃣ Visualize Inference

Bounding boxes for detected vehicles and headlights are displayed on test images.

📊 Results Summary

High detection accuracy in low-light conditions

Significant reduction in headlight glare based on luminance metrics

Near real-time inference performance on GPU

Competitive performance compared to YOLOv5, SSD, and RetinaNet

⚠️ Limitations

Software-based simulation (no real headlight hardware)

Single-frame inference without temporal smoothing

Performance depends on dataset quality

🔮 Future Work

Real-world vehicle integration

Video-based temporal smoothing

Multi-camera fusion

Hardware-in-the-loop testing
