# Adaptive Headlight System Using Faster R-CNN

This project implements a **computer vision–based adaptive headlight system** aimed at reducing glare from oncoming vehicles and improving road visibility during nighttime driving. The system leverages **Faster R-CNN** for vehicle and headlight detection and evaluates performance using detection accuracy, luminance-based glare analysis, and runtime efficiency.

---

## 🚀 Key Features

### Vehicle and Headlight Detection
* Two-wheelers and four-wheelers
* Headlight region identification
* Faster R-CNN with ResNet-50 FPN backbone

### Custom Dataset Support
* XML annotations (Pascal VOC format)
* Train/test split for robust evaluation

### Luminance-Based Glare Analysis
* Measures average headlight luminance
* Software-based glare mitigation

### Quantitative Evaluation Metrics
* Mean Average Precision (mAP), Precision, Recall
* Frame processing time
* Graphical comparisons with YOLOv5, SSD, RetinaNet

### Visualization
* Bounding box inference visualization
* Glare reduction and detection performance graphs

### Platform Flexibility
* Runs on Google Colab or local GPU
* Python, PyTorch, OpenCV, Matplotlib, TorchMetrics

---

## 🧠 Motivation

Nighttime driving is often impaired by **glare from high-beam headlights**, which reduces visibility and increases driver discomfort. Existing adaptive headlight systems rely on **hardware sensors and fixed thresholds**, which are inaccurate in complex traffic scenarios.

This project proposes a **software-based AI solution** that dynamically detects vehicles and headlights using **deep learning**, enabling **realistic adaptive glare mitigation simulations** without hardware limitations.

---

## 🏗️ System Architecture

```
Input Night Image
        ↓
Faster R-CNN Object Detection
        ↓
Headlight Region Identification
        ↓
Luminance Analysis (Glare Measurement)
        ↓
Adaptive Headlight Simulation (Software-Based)
        ↓
Performance Evaluation & Visualization
```

* **Object Detection**: Detects vehicles and headlights using Faster R-CNN  
* **Glare Analysis**: Computes average luminance to estimate glare  
* **Simulation**: Adjusts headlight effect in software for visualization  
* **Evaluation**: Measures accuracy, runtime, and visual performance  

---

## ⚙️ Methodology Overview

### Dataset Structure

```
dataset/
└── split/
    ├── images/
    │   ├── train/
    │   └── test/
    └── annotations/
        ├── train/
        └── test/
```

* Image format: `.jpg`  
* Annotation format: `.xml` (Pascal VOC)  
* Classes:  
  1. Two-wheeler  
  2. Four-wheeler  
  3. Headlight  

### Installation

```bash
pip install torch torchvision torchmetrics opencv-python matplotlib
```

### How to Run

#### 1️⃣ Train the Model

```bash
python main.py
```
* Loads the dataset  
* Trains Faster R-CNN for 10 epochs  
* Saves the model as `faster_rcnn.pth`  

#### 2️⃣ Evaluate the Model

* Computes mAP, precision, recall  
* Measures average headlight luminance  
* Calculates frame processing time  
* Generates comparison graphs  

#### 3️⃣ View Inference Results

* Displays bounding boxes for vehicles and headlights on test images  

---

## 📊 Experimental Evaluation

### Metrics

* **Detection Accuracy**: Mean Average Precision (mAP)  
* **Glare Reduction**: Average luminance in detected headlight regions  
* **Runtime Performance**: Inference time per frame  

### Comparative Analysis

Performance compared against:

* YOLOv5  
* SSD  
* RetinaNet  

Graphs include:

* mAP comparison  
* Luminance reduction comparison  
* Frame processing time comparison  

---

## 🧪 Observations

* Faster R-CNN provides robust detection under low-light conditions  
* Software-based glare mitigation improves visual clarity  
* Feasible for real-time ADAS simulation  

---

## 🔮 Future Work

* Real-time hardware integration with adaptive headlights  
* Temporal smoothing across video frames  
* Multi-camera fusion for wider coverage  
* Real-world vehicle testing  

---

## 👥 Authors

* Barun Gnanasekaran
* Prabu Thangamurugan
