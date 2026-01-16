#!/usr/bin/env python3
"""
Test script for local server testing
Tests the API endpoint with a simple image
"""
import requests
import base64
import cv2
import numpy as np
import json
from pathlib import Path

def create_test_image():
    """Create a simple test image with a white rectangle."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), 3)
    return img

def encode_image(image):
    """Encode image as base64."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')

def test_health(server_url):
    """Test health endpoint."""
    print(f"🔍 Testing health endpoint: {server_url}/health")
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        print(f"✅ Health check passed:")
        print(f"   Status: {health.get('status')}")
        print(f"   Detector ready: {health.get('detector_ready')}")
        print(f"   Analyzer ready: {health.get('analyzer_ready')}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_detection(server_url, image_base64, timestamp):
    """Test detection endpoint."""
    print(f"\n🔍 Testing detection endpoint: {server_url}/detect")
    print(f"   Image size: {len(image_base64)} bytes (base64)")
    print(f"   Timestamp: {timestamp}")
    
    payload = {
        "image_base64": image_base64,
        "timestamp": timestamp
    }
    
    try:
        print("   Sending request (this may take 30-60 seconds)...")
        response = requests.post(
            f"{server_url}/detect",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get('success'):
            detections = result.get('detections', [])
            crops = result.get('crops', {})
            metadata = result.get('metadata', {})
            
            print(f"\n✅ Detection successful!")
            print(f"   Objects detected: {len(detections)}")
            print(f"   Crops generated: {len(crops)}")
            print(f"   Image size: {metadata.get('image_size')}")
            print(f"   Total detections: {metadata.get('total_detections')}")
            print(f"   Useful objects: {metadata.get('useful_objects')}")
            
            # Show first few detections
            if detections:
                print(f"\n   First {min(3, len(detections))} objects:")
                for i, obj in enumerate(detections[:3]):
                    name = obj.get('name', 'Unknown')
                    category = obj.get('category', 'N/A')
                    useful = obj.get('useful', 'N/A')
                    print(f"     {i+1}. {name} ({category}) - useful: {useful}")
            
            return True
        else:
            error = result.get('error', 'Unknown error')
            print(f"❌ Detection failed: {error}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout (server took too long)")
        return False
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False

def main():
    """Main test function."""
    server_url = "http://localhost:8000"
    
    print("="*60)
    print("🧪 Testing 1UP Server API (Local)")
    print("="*60)
    print(f"\nServer URL: {server_url}")
    print("Make sure the server is running: python server/api.py")
    print()
    
    # Test 1: Health check
    if not test_health(server_url):
        print("\n❌ Health check failed. Is the server running?")
        print("   Start server with: cd server && python api.py")
        return
    
    # Test 2: Create test image
    print("\n📸 Creating test image...")
    test_image = create_test_image()
    image_base64 = encode_image(test_image)
    timestamp = "test_20260110_123456"
    
    # Test 3: Detection
    success = test_detection(server_url, image_base64, timestamp)
    
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)

if __name__ == "__main__":
    main()
