# 빠른 시작 가이드

## 1단계: 환경 설정 (5분)

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 패키지 설치
pip install -r requirements.txt
```

## 2단계: 데이터 다운로드 (10-20분)

1. Roboflow 계정 생성: https://app.roboflow.com/
2. API 키 발급: https://app.roboflow.com/settings/api
3. 스크립트에 API 키 입력:
   - `scripts/download_data.py` 파일 열기
   - 9번째 줄의 `"YOUR_API_KEY_HERE"`를 실제 API 키로 변경

4. 데이터 다운로드:
```bash
python scripts/download_data.py
```

5. **데이터 형식 변환 (YOLO → 분류)**:
```bash
python scripts/convert_yolo_to_classification.py
```

## 3단계: 데이터 탐색 (선택사항)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## 4단계: 모델 학습

### 안구 질환 분류 (1-2시간)
```bash
python src/train_eye_classifier.py
```

### 치주 질환 분류 (2-4시간)
```bash
python src/train_dental_classifier.py
```

### 소변 침전물 탐지 (4-6시간)
```bash
python src/train_urine_detector.py
```

💡 **팁**: GPU가 없다면 Google Colab 사용을 권장합니다!

## 5단계: 모델 테스트

```bash
# 안구 질환
python src/inference.py --model eye --image test_images/dog_eye.jpg

# 치주 질환
python src/inference.py --model dental --image test_images/dog_dental.jpg

# 소변 침전물 (결과 이미지 저장)
python src/inference.py --model urine --image test_images/urine.jpg --save result.jpg
```

## 6단계: API 서버 시작

```bash
# 서버 시작
python api/main.py

# 또는
uvicorn api.main:app --reload --port 8000
```

API 문서: http://localhost:8000/docs

## 7단계: API 테스트

```bash
python scripts/test_api.py
```

## 학습 모니터링

```bash
# TensorBoard 실행
tensorboard --logdir runs
```

http://localhost:6006 에서 확인

## 문제 해결

### GPU 메모리 부족
- 배치 크기 줄이기: `BATCH_SIZE = 16` → `8`
- 더 작은 모델 사용: `efficientnet_b2` → `efficientnet_b0`

### 데이터 다운로드 실패
- API 키 확인
- 인터넷 연결 확인
- Roboflow 웹사이트에서 수동 다운로드

### 모델 로드 오류
- 모델 파일 존재 확인: `models/*/best_model.pt`
- 먼저 학습 실행: `python src/train_*.py`

## 다음 단계

- [ ] 하이퍼파라미터 튜닝
- [ ] 모델 앙상블
- [ ] Grad-CAM으로 시각화
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포
