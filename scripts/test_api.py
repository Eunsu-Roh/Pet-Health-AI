"""
API 테스트 스크립트
"""
import requests
import json
from pathlib import Path


def test_endpoint(url, image_path):
    """API 엔드포인트 테스트"""
    print(f"\n📡 테스트 중: {url}")
    print(f"📷 이미지: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 성공!")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(f"❌ 오류 (Status {response.status_code})")
                print(response.text)
        
        except requests.exceptions.ConnectionError:
            print("❌ 서버에 연결할 수 없습니다.")
            print("   먼저 API 서버를 시작하세요: uvicorn api.main:app --reload")
        except Exception as e:
            print(f"❌ 오류: {e}")


def main():
    BASE_URL = "http://localhost:8000"
    
    print("=" * 60)
    print("🧪 Pet Health AI API 테스트")
    print("=" * 60)
    
    # 헬스 체크
    print("\n📊 헬스 체크...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ 서버 정상 작동")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"⚠️  서버 응답 이상 (Status {response.status_code})")
    except:
        print("❌ 서버에 연결할 수 없습니다.")
        print("   먼저 API 서버를 시작하세요: python api/main.py")
        return
    
    # 테스트 이미지 경로 (실제 이미지로 변경 필요)
    test_images = {
        'eye': 'test_images/dog_eye.jpg',
        'dental': 'test_images/dog_dental.jpg',
        'urine': 'test_images/urine_sediment.jpg'
    }
    
    # 안구 질환 테스트
    if Path(test_images['eye']).exists():
        test_endpoint(f"{BASE_URL}/analyze/eye", test_images['eye'])
    else:
        print(f"\n⚠️  안구 테스트 이미지가 없습니다: {test_images['eye']}")
    
    # 치주 질환 테스트
    if Path(test_images['dental']).exists():
        test_endpoint(f"{BASE_URL}/analyze/dental", test_images['dental'])
    else:
        print(f"\n⚠️  치주 테스트 이미지가 없습니다: {test_images['dental']}")
    
    # 소변 침전물 테스트
    if Path(test_images['urine']).exists():
        test_endpoint(f"{BASE_URL}/analyze/urine", test_images['urine'])
    else:
        print(f"\n⚠️  소변 테스트 이미지가 없습니다: {test_images['urine']}")
    
    print("\n" + "=" * 60)
    print("💡 cURL 사용 예제:")
    print("-" * 60)
    print(f"curl -X POST \"{BASE_URL}/analyze/eye\" \\")
    print(f"  -F \"file=@{test_images['eye']}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
