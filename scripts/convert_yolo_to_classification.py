"""
YOLO 객체 탐지 데이터셋을 분류 데이터셋으로 변환
이미지 전체를 해당 클래스 폴더로 복사
"""
import os
import shutil
import yaml
from pathlib import Path
from collections import defaultdict


def convert_yolo_to_classification(data_dir, output_dir):
    """
    YOLO 형식 데이터를 분류 형식으로 변환
    
    Args:
        data_dir: YOLO 데이터 디렉토리 (data.yaml 포함)
        output_dir: 출력 디렉토리
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # data.yaml 읽기
    yaml_path = data_dir / 'data.yaml'
    if not yaml_path.exists():
        print(f"❌ data.yaml을 찾을 수 없습니다: {yaml_path}")
        return False
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    print(f"\n📊 클래스: {class_names}")
    print(f"   클래스 수: {len(class_names)}\n")
    
    # train, valid, test 처리
    splits = ['train', 'valid', 'test']
    total_converted = 0
    
    for split in splits:
        images_dir = data_dir / split / 'images'
        labels_dir = data_dir / split / 'labels'
        
        if not images_dir.exists():
            print(f"⚠️  {split} 이미지 폴더가 없습니다. 건너뜁니다.")
            continue
        
        print(f"📂 {split.upper()} 데이터 변환 중...")
        
        # 출력 디렉토리 생성
        for class_name in class_names:
            class_dir = output_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지별로 처리
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        # 각 이미지의 클래스 확인 (라벨 파일에서)
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # 라벨 파일이 있으면 클래스 읽기
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # 첫 번째 객체의 클래스 사용 (여러 객체 있을 수 있음)
                    first_line = lines[0].strip().split()
                    class_id = int(first_line[0])
                    
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                        
                        # 이미지 복사
                        dest_path = output_dir / split / class_name / img_path.name
                        shutil.copy(img_path, dest_path)
                        total_converted += 1
            else:
                # 라벨이 없는 이미지는 unknown 폴더로
                unknown_dir = output_dir / split / 'unknown'
                unknown_dir.mkdir(parents=True, exist_ok=True)
                dest_path = unknown_dir / img_path.name
                shutil.copy(img_path, dest_path)
        
        print(f"✅ {split} 완료!")
    
    # 통계 출력
    print(f"\n📊 변환 완료 통계:")
    for split in splits:
        split_dir = output_dir / split
        if split_dir.exists():
            print(f"\n  {split.upper()}:")
            for class_name in class_names:
                class_dir = split_dir / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob('*.jpg'))) + len(list(class_dir.glob('*.png')))
                    print(f"    {class_name}: {count}장")
    
    print(f"\n🎉 총 {total_converted}장의 이미지 변환 완료!")
    print(f"📁 저장 위치: {output_dir}")
    return True


def main():
    print("=" * 60)
    print("🔄 YOLO → 분류 데이터셋 변환")
    print("=" * 60)
    
    # 1. 안구 질환 데이터셋 변환
    print("\n[1/2] 안구 질환 데이터셋 변환 중...")
    eye_yolo_dir = Path("data/dog_eye")
    eye_cls_dir = Path("data/dog_eye_classification")
    
    if eye_yolo_dir.exists():
        convert_yolo_to_classification(eye_yolo_dir, eye_cls_dir)
    else:
        print(f"❌ 안구 데이터를 찾을 수 없습니다: {eye_yolo_dir}")
        print("   먼저 python scripts/download_data.py를 실행하세요.")
    
    print("\n" + "-" * 60)
    
    # 2. 치주 질환 데이터셋 변환 (있을 경우)
    print("\n[2/2] 치주 질환 데이터셋 변환 중...")
    dental_yolo_dir = Path("data/dog_dental")
    dental_cls_dir = Path("data/dog_dental_classification")
    
    if dental_yolo_dir.exists():
        # folder 형식인지 yolo 형식인지 확인
        if (dental_yolo_dir / 'data.yaml').exists():
            # YOLO 형식
            convert_yolo_to_classification(dental_yolo_dir, dental_cls_dir)
        elif (dental_yolo_dir / 'train').exists():
            # 이미 분류 형식
            print("✅ 치주 데이터는 이미 분류 형식입니다!")
            print(f"   위치: {dental_yolo_dir}")
        else:
            print(f"⚠️  치주 데이터 형식을 확인할 수 없습니다.")
    else:
        print(f"⚠️  치주 데이터를 찾을 수 없습니다: {dental_yolo_dir}")
    
    print("\n" + "=" * 60)
    print("🎉 변환 완료!")
    print("=" * 60)
    print("\n다음 단계:")
    print("1. 안구 분류 모델 학습:")
    print("   python src/train_eye_classifier.py --data data/dog_eye_classification")
    print("\n2. 치주 분류 모델 학습:")
    print("   python src/train_dental_classifier.py --data data/dog_dental_classification")
    print("\n3. 소변 탐지 모델 학습 (YOLO):")
    print("   python src/train_urine_detector.py")


if __name__ == "__main__":
    main()
