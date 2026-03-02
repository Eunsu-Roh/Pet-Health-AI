"""
모델 추론 스크립트
"""
import torch
import torchvision.models as models
from torch import nn
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO


class ImageClassifier:
    """이미지 분류 모델 추론"""
    
    def __init__(self, model_path, num_classes, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        
        # 모델 로드
        self.model = models.efficientnet_b2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def predict(self, image_path):
        """이미지 분류 예측"""
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 전처리
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # 모든 클래스별 확률
        all_probs = {
            self.class_names[i]: probabilities[0][i].item() 
            for i in range(len(self.class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': all_probs
        }


class ObjectDetector:
    """객체 탐지 모델 추론 (YOLOv8)"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path, conf_threshold=0.5):
        """객체 탐지 예측"""
        results = self.model.predict(
            source=str(image_path),
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return {
            'num_detections': len(detections),
            'detections': detections,
            'image_shape': results[0].orig_shape
        }
    
    def visualize(self, image_path, save_path=None):
        """탐지 결과 시각화"""
        results = self.model.predict(source=str(image_path), conf=0.5)
        
        # 결과 이미지 가져오기
        annotated_img = results[0].plot()
        
        if save_path:
            cv2.imwrite(str(save_path), annotated_img)
            print(f"✅ 결과 이미지 저장: {save_path}")
        
        return annotated_img


def main():
    parser = argparse.ArgumentParser(description='모델 추론')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['eye', 'dental', 'urine'],
                       help='모델 종류')
    parser.add_argument('--image', type=str, required=True,
                       help='입력 이미지 경로')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='신뢰도 임계값 (객체 탐지용)')
    parser.add_argument('--save', type=str, default=None,
                       help='결과 저장 경로 (객체 탐지용)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🔍 {args.model.upper()} 모델 추론")
    print("=" * 60)
    print(f"📷 이미지: {args.image}\n")
    
    if not Path(args.image).exists():
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        return
    
    if args.model == 'eye':
        # 안구 질환 분류
        model_path = "models/eye_classifier/best_model.pt"
        # 실제 클래스명은 데이터 변환 후 확인 필요
        class_names = ['class_0', 'class_1', 'class_2']  # 실제 클래스명으로 수정 필요
        
        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            print("   먼저 모델을 학습하세요: python src/train_eye_classifier.py")
            return
        
        classifier = ImageClassifier(model_path, len(class_names), class_names)
        result = classifier.predict(args.image)
        
        print("📊 예측 결과:")
        print(f"   클래스: {result['predicted_class']}")
        print(f"   신뢰도: {result['confidence']*100:.2f}%")
        print("\n📈 모든 클래스별 확률:")
        for cls, prob in result['all_probabilities'].items():
            print(f"   {cls}: {prob*100:.2f}%")
    
    elif args.model == 'dental':
        # 치주 질환 분류
        model_path = "models/dental_classifier/best_model.pt"
        # 실제 클래스명은 데이터 변환 후 확인 필요
        class_names = ['class_0', 'class_1', 'class_2']  # 실제 클래스명으로 수정
        
        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            print("   먼저 모델을 학습하세요: python src/train_dental_classifier.py")
            return
        
        classifier = ImageClassifier(model_path, len(class_names), class_names)
        result = classifier.predict(args.image)
        
        print("📊 예측 결과:")
        print(f"   치석 단계: {result['predicted_class']}")
        print(f"   신뢰도: {result['confidence']*100:.2f}%")
        print("\n📈 모든 단계별 확률:")
        for cls, prob in result['all_probabilities'].items():
            print(f"   {cls}: {prob*100:.2f}%")
    
    elif args.model == 'urine':
        # 소변 침전물 탐지
        model_path = "models/urine_detector/best_model.pt"
        
        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            print("   먼저 모델을 학습하세요: python src/train_urine_detector.py")
            return
        
        detector = ObjectDetector(model_path)
        result = detector.predict(args.image, conf_threshold=args.conf)
        
        print(f"📊 탐지 결과: {result['num_detections']}개 객체 발견")
        print(f"   이미지 크기: {result['image_shape']}\n")
        
        for i, det in enumerate(result['detections'], 1):
            print(f"   [{i}] {det['class']}")
            print(f"       신뢰도: {det['confidence']*100:.2f}%")
            print(f"       위치: {det['bbox']}")
        
        # 시각화
        if args.save:
            detector.visualize(args.image, args.save)
        else:
            print("\n💡 결과 이미지를 저장하려면 --save 옵션을 사용하세요")
            print("   예: --save result.jpg")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
