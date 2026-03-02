"""
소변 침전물 객체 탐지 모델 학습
YOLOv8 사용
"""
from ultralytics import YOLO
from pathlib import Path
import yaml


def main():
    # 설정
    DATA_DIR = Path("data/urine_sediment")
    SAVE_DIR = Path("models/urine_detector")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # YOLOv8 모델 크기 선택
    # yolov8n: nano (가장 빠름, 작음)
    # yolov8s: small
    # yolov8m: medium
    # yolov8l: large
    # yolov8x: xlarge (가장 정확, 느림)
    MODEL_SIZE = 'yolov8s'  # Small 모델 사용
    
    print("=" * 60)
    print("🔬 소변 침전물 객체 탐지 모델 학습 시작")
    print("=" * 60)
    print(f"📦 모델: {MODEL_SIZE}.pt")
    print(f"📁 데이터: {DATA_DIR}\n")
    
    # data.yaml 파일 경로 확인
    data_yaml = DATA_DIR / "data.yaml"
    
    if not data_yaml.exists():
        print("⚠️  data.yaml 파일을 찾을 수 없습니다!")
        print(f"   예상 위치: {data_yaml}")
        print("\n해결 방법:")
        print("1. Roboflow에서 데이터 다운로드: python scripts/download_data.py")
        print("2. YOLOv8 형식으로 다운로드했는지 확인")
        return
    
    # data.yaml 내용 확인
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print("📊 데이터셋 정보:")
    print(f"   클래스 수: {data_config.get('nc', 'N/A')}")
    print(f"   클래스: {data_config.get('names', 'N/A')}\n")
    
    # YOLOv8 모델 로드
    print(f"🔨 {MODEL_SIZE} 모델 로딩 중...")
    model = YOLO(f'{MODEL_SIZE}.pt')
    print("✅ 모델 준비 완료\n")
    
    # 학습 시작
    print("🚀 학습 시작!\n")
    results = model.train(
        data=str(data_yaml),
        epochs=50,              # 에포크 수
        imgsz=640,              # 이미지 크기
        batch=16,               # 배치 크기 (GPU 메모리에 따라 조정)
        name='urine_sediment',  # 실험 이름
        patience=10,            # Early stopping patience
        save=True,              # 모델 저장
        plots=True,             # 학습 그래프 저장
        device=0,               # GPU 사용 (CPU는 'cpu')
        
        # 추가 설정
        optimizer='AdamW',
        lr0=0.001,              # 초기 학습률
        lrf=0.01,               # 최종 학습률 (lr0 * lrf)
        warmup_epochs=3,        # Warmup 에포크
        
        # 데이터 증강
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
        degrees=0.0,            # 회전 (±deg)
        translate=0.1,          # 이동 (±fraction)
        scale=0.5,              # 스케일 (gain)
        flipud=0.0,             # 상하 반전
        fliplr=0.5,             # 좌우 반전
        mosaic=1.0,             # Mosaic augmentation
    )
    
    print("\n" + "=" * 60)
    print("🎉 학습 완료!")
    print("=" * 60)
    
    # 검증
    print("\n📊 모델 검증 중...")
    metrics = model.val()
    
    print("\n📈 성능 지표:")
    print(f"   mAP@50: {metrics.box.map50:.4f}")
    print(f"   mAP@50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    # 모델 저장
    best_model_path = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    final_model_path = SAVE_DIR / 'best_model.pt'
    
    # 모델 복사
    import shutil
    if best_model_path.exists():
        shutil.copy(best_model_path, final_model_path)
        print(f"\n✅ 최고 성능 모델 저장: {final_model_path}")
    
    print(f"\n📁 학습 결과 위치: {model.trainer.save_dir}")
    print("   - weights/best.pt: 최고 성능 모델")
    print("   - weights/last.pt: 최종 모델")
    print("   - 다양한 학습 그래프 및 결과 이미지")
    print("\n💡 TensorBoard 실행: tensorboard --logdir runs")
    print("=" * 60)


if __name__ == "__main__":
    main()
