"""
FastAPI 서버 - 반려동물 건강 분석 API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import torchvision.models as models
from torch import nn
from PIL import Image
import cv2
import numpy as np
import io
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import tempfile
import shutil


# FastAPI 앱 생성
app = FastAPI(
    title="Pet Health AI API",
    description="반려동물 건강 분석 API - 안구/치주 질환 분류, 소변 침전물 탐지",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 응답 모델
class ClassificationResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    message: Optional[str] = None


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    success: bool
    num_detections: int
    detections: List[Detection]
    image_shape: List[int]
    message: Optional[str] = None


# 모델 로더 클래스
class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eye_model = None
        self.dental_model = None
        self.urine_model = None
        
        # Transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # 클래스 이름 (실제 학습 후 수정 필요)
        self.eye_classes = ['class_0', 'class_1']  # 데이터 변환 후 확인하여 수정
        self.dental_classes = ['class_0', 'class_1', 'class_2']  # 데이터 변환 후 확인하여 수정
    
    def load_eye_model(self):
        """안구 질환 모델 로드"""
        if self.eye_model is None:
            model_path = Path("models/eye_classifier/best_model.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"안구 모델을 찾을 수 없습니다: {model_path}")
            
            self.eye_model = models.efficientnet_b2(weights=None)
            in_features = self.eye_model.classifier[1].in_features
            self.eye_model.classifier[1] = nn.Linear(in_features, len(self.eye_classes))
            self.eye_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.eye_model.to(self.device)
            self.eye_model.eval()
        return self.eye_model
    
    def load_dental_model(self):
        """치주 질환 모델 로드"""
        if self.dental_model is None:
            model_path = Path("models/dental_classifier/best_model.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"치주 모델을 찾을 수 없습니다: {model_path}")
            
            self.dental_model = models.efficientnet_b2(weights=None)
            in_features = self.dental_model.classifier[1].in_features
            self.dental_model.classifier[1] = nn.Linear(in_features, len(self.dental_classes))
            self.dental_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.dental_model.to(self.device)
            self.dental_model.eval()
        return self.dental_model
    
    def load_urine_model(self):
        """소변 침전물 모델 로드"""
        if self.urine_model is None:
            model_path = Path("models/urine_detector/best_model.pt")
            if not model_path.exists():
                raise FileNotFoundError(f"소변 모델을 찾을 수 없습니다: {model_path}")
            
            self.urine_model = YOLO(str(model_path))
        return self.urine_model


# 전역 모델 로더
model_loader = ModelLoader()


def preprocess_image(image_bytes):
    """이미지 전처리"""
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image.convert('RGB'))
    return image


@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "Pet Health AI API",
        "version": "1.0.0",
        "endpoints": {
            "eye": "/analyze/eye",
            "dental": "/analyze/dental",
            "urine": "/analyze/urine",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "device": str(model_loader.device),
        "models": {
            "eye": model_loader.eye_model is not None,
            "dental": model_loader.dental_model is not None,
            "urine": model_loader.urine_model is not None
        }
    }


@app.post("/analyze/eye", response_model=ClassificationResponse)
async def analyze_eye(file: UploadFile = File(...)):
    """
    안구 질환 분류
    
    - **file**: 개 안구 이미지 (JPG, PNG)
    """
    try:
        # 모델 로드
        model = model_loader.load_eye_model()
        
        # 이미지 전처리
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        transformed = model_loader.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(model_loader.device)
        
        # 추론
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = model_loader.eye_classes[predicted.item()]
        confidence_score = confidence.item()
        
        all_probs = {
            model_loader.eye_classes[i]: float(probabilities[0][i].item())
            for i in range(len(model_loader.eye_classes))
        }
        
        return ClassificationResponse(
            success=True,
            predicted_class=predicted_class,
            confidence=confidence_score,
            all_probabilities=all_probs,
            message="안구 질환 분석 완료"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/analyze/dental", response_model=ClassificationResponse)
async def analyze_dental(file: UploadFile = File(...)):
    """
    치주 질환 분류
    
    - **file**: 개 치아 이미지 (JPG, PNG)
    """
    try:
        model = model_loader.load_dental_model()
        
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)
        
        transformed = model_loader.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(model_loader.device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = model_loader.dental_classes[predicted.item()]
        confidence_score = confidence.item()
        
        all_probs = {
            model_loader.dental_classes[i]: float(probabilities[0][i].item())
            for i in range(len(model_loader.dental_classes))
        }
        
        return ClassificationResponse(
            success=True,
            predicted_class=predicted_class,
            confidence=confidence_score,
            all_probabilities=all_probs,
            message="치주 질환 분석 완료"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


@app.post("/analyze/urine", response_model=DetectionResponse)
async def analyze_urine(
    file: UploadFile = File(...),
    conf_threshold: float = 0.5
):
    """
    소변 침전물 객체 탐지
    
    - **file**: 소변 현미경 이미지 (JPG, PNG)
    - **conf_threshold**: 신뢰도 임계값 (0.0~1.0)
    """
    try:
        model = model_loader.load_urine_model()
        
        # 임시 파일로 저장 (YOLO는 파일 경로 필요)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image_bytes = await file.read()
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        # 추론
        results = model.predict(
            source=tmp_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        # 결과 파싱
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = Detection(
                    class_name=result.names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=[float(x) for x in box.xyxy[0].tolist()]
                )
                detections.append(detection)
        
        # 임시 파일 삭제
        Path(tmp_path).unlink()
        
        return DetectionResponse(
            success=True,
            num_detections=len(detections),
            detections=detections,
            image_shape=list(results[0].orig_shape),
            message="소변 침전물 분석 완료"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Pet Health AI API 서버 시작")
    print("=" * 60)
    print("📡 서버 주소: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
