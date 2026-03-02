"""
Roboflow에서 데이터셋 다운로드
사용 전 Roboflow API 키 설정 필요: https://app.roboflow.com/settings/api
"""
import os
from roboflow import Roboflow

# Roboflow API 키 설정 (환경변수 또는 직접 입력)
# 아래 줄의 "YOUR_API_KEY_HERE"를 실제 API 키로 변경하세요
API_KEY = os.getenv("ROBOFLOW_API_KEY", "9ACAbai7kXxXBNC2mgic")

if API_KEY == "YOUR_API_KEY_HERE":
    print("⚠️  Roboflow API 키를 설정해주세요!")
    print("1. https://app.roboflow.com/settings/api 에서 API 키 발급")
    print("2. 환경변수 설정: set ROBOFLOW_API_KEY=your_api_key")
    print("   또는 이 파일의 API_KEY 변수에 직접 입력")
    exit(1)

rf = Roboflow(api_key=API_KEY)

print("🐕 반려동물 Healthcare 데이터셋 다운로드 시작...\n")

# 1. 안구 질환 데이터셋 (Object Detection - YOLOv8 형식으로 다운로드)
print("📥 [1/3] 안구 질환 데이터셋 다운로드 중...")
try:
    eye_project = rf.workspace("jonathan-chandra").project("dog-eye-problems-detection")
    eye_dataset = eye_project.version(1).download("yolov8", location="data/dog_eye")
    print("✅ 안구 질환 데이터셋 다운로드 완료! (YOLOv8 형식)\n")
except Exception as e:
    print(f"❌ 안구 데이터셋 다운로드 실패: {e}\n")

# 2. 치주 질환 데이터셋 (YOLOv8 형식으로 시도)
print("📥 [2/3] 치주 질환 데이터셋 다운로드 중...")
try:
    dental_project = rf.workspace("precogmed").project("dentalumeshu")
    # 먼저 folder 형식 시도, 실패하면 yolov8
    try:
        dental_dataset = dental_project.version(1).download("folder", location="data/dog_dental")
        print("✅ 치주 질환 데이터셋 다운로드 완료! (Folder 형식)\n")
    except:
        dental_dataset = dental_project.version(1).download("yolov8", location="data/dog_dental")
        print("✅ 치주 질환 데이터셋 다운로드 완료! (YOLOv8 형식)\n")
except Exception as e:
    print(f"❌ 치주 데이터셋 다운로드 실패: {e}\n")

# 3. 소변 침전물 데이터셋 (YOLOv8 형식)
print("📥 [3/3] 소변 침전물 데이터셋 다운로드 중...")
try:
    urine_project = rf.workspace("object-detection-6qhgz").project("urine-sediment-detection")
    urine_dataset = urine_project.version(1).download("yolov8", location="data/urine_sediment")
    print("✅ 소변 침전물 데이터셋 다운로드 완료!\n")
except Exception as e:
    print(f"❌ 소변 데이터셋 다운로드 실패: {e}\n")

print("=" * 60)
print("🎉 모든 데이터셋 다운로드 완료!")
print("=" * 60)
print("\n다음 단계:")
print("1. 데이터 탐색: jupyter notebook notebooks/01_data_exploration.ipynb")
print("2. 모델 학습: python src/train_eye_classifier.py")
