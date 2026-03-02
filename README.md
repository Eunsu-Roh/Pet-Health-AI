# Pet Health AI - 반려동물 Healthcare 전체 파이프라인 실습

반려동물 건강 관리를 위한 AI 모델 학습 및 배포 전체 파이프라인 실습 프로젝트

## 📋 프로젝트 개요

### 구현 모델
1. **안구 질환 분류** (Image Classification)
   - 모델: EfficientNet-B2
   - 클래스: 정상, 결막염, 백내장 등
   - 데이터: Roboflow Dog Eye Problems

2. **치주 질환 분류** (Image Classification)
   - 모델: EfficientNet-B2
   - 클래스: 정상, 치석 1~4단계
   - 데이터: Roboflow DentalUmeshu (~9,490장)

3. **소변 침전물 탐지** (Object Detection)
   - 모델: YOLOv8
   - 클래스: 백혈구, 적혈구, 결정체, 상피세포 등 14개
   - 데이터: Roboflow Urine Sediment (~5,700장)

## 🚀 시작하기

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 다운로드
```bash
# Roboflow API 키 설정 (scripts/download_data.py 파일에 직접 입력)
python scripts/download_data.py

# YOLO 데이터를 분류 형식으로 변환
python scripts/convert_yolo_to_classification.py
```

### 3. 모델 학습
```bash
# 안구 질환 분류
python src/train_eye_classifier.py

# 치주 질환 분류
python src/train_dental_classifier.py

# 소변 침전물 탐지
python src/train_urine_detector.py
```

### 4. API 서버 시작
```bash
uvicorn api.main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

### 5. 추론 테스트
```bash
python src/inference.py --model eye --image test_images/dog_eye.jpg
```

## 📁 프로젝트 구조
```
pet-health-ai/
├── data/                     # 데이터셋 (다운로드 후 생성)
│   ├── dog_eye/
│   ├── dog_dental/
│   └── urine_sediment/
├── models/                   # 학습된 모델 저장
│   ├── eye_classifier/
│   ├── dental_classifier/
│   └── urine_detector/
├── notebooks/                # Jupyter 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_eye_training.ipynb
│   └── 03_dental_training.ipynb
├── src/                      # 소스 코드
│   ├── train_eye_classifier.py
│   ├── train_dental_classifier.py
│   ├── train_urine_detector.py
│   ├── inference.py
│   ├── dataset.py
│   └── utils.py
├── api/                      # FastAPI 서버
│   └── main.py
├── scripts/                  # 유틸리티 스크립트
│   └── download_data.py
├── test_images/              # 테스트용 이미지
├── requirements.txt
└── README.md
```

## 📊 성능 목표

| 모델 | 목표 Accuracy/mAP | 학습 시간 (예상) |
|------|------------------|-----------------|
| 안구 분류 | 85%+ | 1-2시간 |
| 치주 분류 | 88%+ | 2-4시간 |
| 소변 탐지 | mAP@50 65%+ | 4-6시간 |

## 🛠️ 기술 스택

- **Deep Learning**: PyTorch, torchvision
- **Computer Vision**: OpenCV, Albumentations
- **Object Detection**: Ultralytics YOLOv8
- **API Framework**: FastAPI
- **Visualization**: Matplotlib, Seaborn, TensorBoard

## 📝 학습 내용

이 프로젝트를 통해 다음을 경험합니다:
- [x] 전이학습 (Transfer Learning)
- [x] 이미지 분류 (Image Classification)
- [x] 객체 탐지 (Object Detection)
- [x] 데이터 증강 (Data Augmentation)
- [x] REST API 구축
- [x] 모델 배포 파이프라인

## 📄 라이선스

학습용 프로젝트입니다. 데이터셋은 각각의 라이선스를 따릅니다.
