# Pet Health AI - End-to-End Pet Healthcare Pipeline

A hands-on project covering the full pipeline from AI model training to deployment for companion animal healthcare management.

## 📋 Project Overview

### Implemented Models
1. **Ocular Disease Classification** (Image Classification)
   - **Model**: EfficientNet-B2
   - **Classes**: Normal, Conjunctivitis, Cataracts, etc.
   - **Dataset**: Roboflow Dog Eye Problems

2. **Periodontal Disease Classification** (Image Classification)
   - **Model**: EfficientNet-B2
   - **Classes**: Normal, Calculus Stages 1–4
   - **Dataset**: Roboflow DentalUmeshu (~9,490 images)

3. **Urine Sediment Detection** (Object Detection)
   - **Model**: YOLOv8
   - **Classes**: 14 classes including WBC, RBC, Crystals, Epithelial Cells, etc.
   - **Dataset**: Roboflow Urine Sediment (~5,700 images)

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create a virtual environment (Recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install packages
pip install -r requirements.txt
```

### 2. Download Data
```bash
# Set Roboflow API key (Enter directly in scripts/download_data.py)
python scripts/download_data.py

# Convert YOLO data format to classification format
python scripts/convert_yolo_to_classification.py
```

### 3. Model Training
```bash
# Ocular disease classification
python src/train_eye_classifier.py

# Periodontal disease classification
python src/train_dental_classifier.py

# Urine sediment detection
python src/train_urine_detector.py
```

### 4. Start API Server
```bash
uvicorn api.main:app --reload --port 8000
```

API Documentation: http://localhost:8000/docs

### 5. Run Inference Test
```bash
python src/inference.py --model eye --image test_images/dog_eye.jpg
```

---

## 📁 Project Structure

```text
pet-health-ai/
├── data/                      # Datasets (generated after download)
│   ├── dog_eye/
│   ├── dog_dental/
│   └── urine_sediment/
├── models/                    # Saved trained models
│   ├── eye_classifier/
│   ├── dental_classifier/
│   └── urine_detector/
├── notebooks/                 # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_eye_training.ipynb
│   └── 03_dental_training.ipynb
├── src/                       # Source code
│   ├── train_eye_classifier.py
│   ├── train_dental_classifier.py
│   ├── train_urine_detector.py
│   ├── inference.py
│   ├── dataset.py
│   └── utils.py
├── api/                       # FastAPI server
│   └── main.py
├── scripts/                   # Utility scripts
│   └── download_data.py
├── test_images/               # Test sample images
├── requirements.txt
└── README.md
```

---

## 📊 Performance Targets

| Model | Target Metric | Estimated Training Time |
|-------|---------------|-------------------------|
| Ocular Classification | Accuracy 85%+ | 1–2 hours |
| Periodontal Classification | Accuracy 88%+ | 2–4 hours |
| Urine Detection | mAP@50 65%+ | 4–6 hours |

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **Computer Vision**: OpenCV, Albumentations
- **Object Detection**: Ultralytics YOLOv8
- **API Framework**: FastAPI
- **Visualization**: Matplotlib, Seaborn, TensorBoard

---

## 📝 Key Takeaways & Practices

Through this project, the following core topics are explored and implemented:
- [x] Transfer Learning
- [x] Image Classification
- [x] Object Detection
- [x] Data Augmentation
- [x] REST API Construction
- [x] Model Deployment Pipeline

---

## 📄 License

This repository is intended for educational purposes. Each dataset is subject to its respective original license.
