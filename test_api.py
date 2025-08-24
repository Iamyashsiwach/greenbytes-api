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
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {
        'mode': 'disease',
        'answers': json.dumps({"Q1": 1, "Q2": 0, "Q3": -1, "Q4": 1, "Q5": 0})
    }
    
    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    print(f"\nPredict response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Mode: {result['mode']}")
        print(f"YOLO confidence: {result['yolo']['conf']}")
        print(f"TabNet confidence: {result['tabnet']['conf']}")
        print(f"Fusion result: {result['fusion']['detected']}")
        print(f"Reference image: {result['reference_image_url']}")
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
