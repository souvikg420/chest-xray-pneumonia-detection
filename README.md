# chest-xray-pneumonia-detection
Deep Learning system for Pneumonia detection from X-rays

# 🫁 Chest X-Ray Pneumonia Detection
### End-to-End Medical Image Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models & Results](#models--results)
- [Grad-CAM Explainability](#grad-cam-explainability)
- [API Endpoints](#api-endpoints)
- [Run Locally](#run-locally)
- [Run with Docker](#run-with-docker)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## 🔍 Overview

A production-ready deep learning system that classifies chest X-ray images
as **Normal** or **Pneumonia** using transfer learning with ResNet50 and
EfficientNetB0. The system includes visual explanations via **Grad-CAM**,
a **Flask REST API**, and full **Docker** containerization for deployment.

### Key Features
- ✅ Transfer Learning with ResNet50 & EfficientNetB0
- ✅ Two-phase fine-tuning strategy
- ✅ Class imbalance handling with weighted loss
- ✅ Grad-CAM visual explanations
- ✅ Threshold optimization for medical use
- ✅ Production-ready Flask REST API
- ✅ Dockerized for cloud deployment

---

## 📁 Dataset

**Chest X-Ray Images (Pneumonia)**
- Source: Kaggle / Google Drive
- Total Images: ~5,800
- Classes: NORMAL | PNEUMONIA

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

> Class imbalance ratio: 2.9:1 (Pneumonia:Normal)
> Fixed using: Computed class weights during training

---

## 📂 Project Structure
```
chest-xray-pneumonia-detection/
│
├── src/
│   └── app.py                   # Flask REST API
│
├── models/
│   └── best_model_final.keras   # Trained ResNet50 model
│
├── outputs/
│   ├── sample_images.png
│   ├── class_imbalance.png
│   ├── baseline_training.png
│   ├── resnet_phaseA.png
│   ├── resnet_phaseB.png
│   ├── effnet_phaseA.png
│   ├── effnet_phaseB.png
│   ├── baseline_cnn_eval.png
│   ├── resnet50_eval.png
│   ├── efficientnetb0_eval.png
│   ├── model_comparison.png
│   └── gradcam_ResNet50.png
│
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🏆 Models & Results

### Architecture Overview

#### 1. Baseline CNN (from scratch)
```
Input (224×224×3)
→ Conv2D(32) → BatchNorm → MaxPool → Dropout
→ Conv2D(64) → BatchNorm → MaxPool → Dropout
→ Conv2D(128) → BatchNorm → MaxPool → Dropout
→ Conv2D(256) → BatchNorm → MaxPool → Dropout
→ GlobalAveragePooling
→ Dense(512) → Dropout
→ Dense(1, sigmoid)
Parameters: 522,433
```

#### 2. ResNet50 (Transfer Learning)
```
Backbone  : ResNet50 pretrained on ImageNet (25.6M params)
Phase A   : Freeze backbone → Train head only (LR=1e-4, 10 epochs)
Phase B   : Unfreeze last 35 layers → Fine-tune (LR=1e-5, 20 epochs)
Head      : GAP → Dense(512) → BN → Dropout → Dense(256) → Sigmoid
```

#### 3. EfficientNetB0 (Transfer Learning — Fixed)
```
Backbone  : EfficientNetB0 pretrained on ImageNet (4.4M params)
Fix       : Conv layers frozen, BatchNorm layers active (domain adaptation)
Phase B   : Full fine-tuning with LR=1e-5
Head      : GAP → Dense(256) → BN → Dropout → Sigmoid
```

---

### 📊 Test Set Results (624 images)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Baseline CNN | 62.50% | 0.6250 | 1.0000 | 0.7692 | 0.498 |
| **ResNet50 Fine-Tuned** | **84.13%** | **0.9008** | 0.8385 | **0.8685** | **0.923** |
| EfficientNetB0 | 84.78% | 0.8471 | **0.9231** | 0.8834 | 0.908 |

### 🏥 False Negative Analysis (Missed Pneumonia Cases)

| Model | Missed Cases | Miss Rate |
|-------|-------------|-----------|
| Baseline CNN | 0 / 390 | 0.0% |
| ResNet50 Fine-Tuned | 63 / 390 | 16.2% |
| EfficientNetB0 | 30 / 390 | 7.7% |

> **In medical AI, Recall is the most critical metric.**
> A False Negative means a sick patient is sent home without treatment.

### 🎯 Threshold Optimization (ResNet50)

| Threshold | Accuracy | Recall | F1 | FN |
|-----------|----------|--------|----|----|
| 0.50 (default) | 84.13% | 83.85% | 0.8685 | 63 |
| **0.30 (optimal)** | **85.74%** | **89.23%** | **0.8866** | **42** |

> Lowering threshold from 0.50 → 0.30 saves 21 additional patients

---

## 🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights which
regions of the X-ray the model focused on when making its prediction.
```
Blue regions  → Model ignored these areas
Red regions   → Model focused here for prediction
```

**How it works:**
1. Forward pass → get prediction score
2. Compute gradients of score w.r.t. last conv layer
3. Global average pool the gradients → importance weights
4. Weighted sum of feature maps → raw heatmap
5. ReLU + Normalize + Resize → overlay on original image

---

## 📡 API Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| GET | `/health` | Health check | API status |
| GET | `/model_info` | Model metadata | Name, params, classes |
| POST | `/predict` | Classify X-ray | Prediction + confidence |

### Request Format
```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<chest_xray_image>
```

### Response Format
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9749,
  "probabilities": {
    "NORMAL": 0.0251,
    "PNEUMONIA": 0.9749
  },
  "time_ms": 93.75
}
```

### Test the API
```bash
# Health check
curl http://localhost:5000/health

# Model info
curl http://localhost:5000/model_info

# Predict from image
curl -X POST http://localhost:5000/predict \
     -F "file=@chest_xray.jpg"
```

---

## 💻 Run Locally

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection

# Install dependencies
pip install -r requirements.txt

# Run API
python src/app.py
```

API will be available at: `http://localhost:5000`

---

## 🐳 Run with Docker

### Build & Run
```bash
# Build image
docker build -t xray-pneumonia-api .

# Run container
docker run -d \
  --name xray-api \
  -p 5000:5000 \
  xray-pneumonia-api

# Check running
docker ps

# View logs
docker logs -f xray-api
```

### Stop & Remove
```bash
docker stop xray-api
docker rm xray-api
```

---

## ☁️ Cloud Deployment

### AWS EC2
```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Install Docker
sudo apt update && sudo apt install -y docker.io

# Build & Run
docker build -t xray-api .
docker run -d -p 5000:5000 --restart always xray-api

# Test
curl http://YOUR_EC2_IP:5000/health
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow 2.15, Keras |
| Models | ResNet50, EfficientNetB0 |
| Explainability | Grad-CAM |
| API | Flask 3.0, Flask-CORS |
| Deployment | Docker, AWS EC2 |
| Data Processing | NumPy, OpenCV, Pillow |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## 📈 Training Strategy
```
Data Augmentation:
  Rotation ±15°, Horizontal flip, Zoom ±10%
  Brightness [0.8-1.2], Shear ±10%, Shift ±10%

Class Imbalance:
  Computed class weights → NORMAL: 1.94, PNEUMONIA: 0.67

Callbacks:
  EarlyStopping    → patience=5, monitor=val_loss
  ReduceLROnPlateau → factor=0.5, patience=3
  ModelCheckpoint  → save best val_auc only

Two-Phase Transfer Learning:
  Phase A: Frozen backbone, LR=1e-4, 10 epochs
  Phase B: Unfreeze last 35 layers, LR=1e-5, 20 epochs
```

---

## ⚠️ Disclaimer

> This system is developed for **research and educational purposes only**.
> It is **NOT** a certified medical device and should **NOT** be used
> for clinical diagnosis without validation by qualified medical professionals.
> Always consult a licensed radiologist for medical decisions.

---

## 👤 Author

Souvik GHosh
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/souvik-ghosh-83a548331)

---


⭐ **If this project helped you, please give it a star!** ⭐
