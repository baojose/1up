#!/usr/bin/env python3
"""Simple test script to verify Python is working."""
import sys

print("="*60, flush=True)
print("TEST: Python is working!", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"Python executable: {sys.executable}", flush=True)
print("="*60, flush=True)

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}", flush=True)
except ImportError as e:
    print(f"❌ OpenCV not found: {e}", flush=True)

try:
    import yaml
    print("✅ PyYAML loaded", flush=True)
except ImportError as e:
    print(f"❌ PyYAML not found: {e}", flush=True)

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}", flush=True)
except ImportError as e:
    print(f"❌ PyTorch not found: {e}", flush=True)

print("="*60, flush=True)
print("Test complete!", flush=True)
