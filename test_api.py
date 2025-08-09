"""
Quick test script to verify the backend API is working
"""

import requests
import json
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Health check: {response.json()}")
    assert response.status_code == 200
    assert response.json()["ok"] == True

def test_predict():
    """Test predict endpoint with stub data"""
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test data
    files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'mode': 'disease',
        'answers': json.dumps([1, 0, -1, 1, 0, 1, -1, 0, 1, 0])
    }
    
    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    print(f"\nPredict response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Mode: {result['mode']}")
        print(f"YOLO confidence: {result['yolo']['conf']}")
        print(f"TabNet probability: {result['tabnet']['proba']}")
        print(f"Fusion result: {result['fusion']['present']}")
        print(f"Reference image: {result['ref_img']}")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing GreenBytes Backend API...")
    print("=" * 50)
    
    try:
        test_health()
        test_predict()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
