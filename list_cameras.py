"""
Simple script to list available cameras
Works even if SAM 3 and other dependencies are not installed.
"""
import sys
import platform

try:
    import cv2
except ImportError:
    print("❌ OpenCV no está instalado")
    print("   Instala con: pip3 install opencv-python")
    sys.exit(1)

print("🔍 Buscando cámaras disponibles...")
print("=" * 60)

available = []
max_index = 10

# Use correct backend for macOS to avoid OpenCV error messages
is_mac = platform.system() == 'Darwin'
backend = cv2.CAP_AVFOUNDATION if is_mac else None

for i in range(max_index):
    # Use backend from the start to avoid OpenCV trying multiple backends
    if backend:
        cap = cv2.VideoCapture(i, backend)
    else:
        cap = cv2.VideoCapture(i)
    
    if cap.isOpened():
        # Try to read a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            available.append({
                'index': i,
                'width': width,
                'height': height,
                'fps': fps
            })
            
            print(f"✅ Cámara {i}: {width}x{height} @ {fps:.1f}fps")
        else:
            print(f"⚠️  Cámara {i}: Abierta pero no puede leer frames")
        
        cap.release()
    else:
        print(f"❌ Cámara {i}: No disponible")

print("=" * 60)

if available:
    print(f"\n📊 Resumen: {len(available)} cámara(s) disponible(s)")
    
    # Suggest external camera
    external = [c for c in available if c['index'] != 0]
    if external:
        best = max(external, key=lambda c: c['width'] * c['height'])
        print(f"\n💡 Recomendación: Usar cámara {best['index']} (externa, {best['width']}x{best['height']})")
        print(f"   Edita config.yaml: camera.index: {best['index']}")
    elif available:
        print(f"\n💡 Usar cámara {available[0]['index']} (única disponible)")
        print(f"   Edita config.yaml: camera.index: {available[0]['index']}")
else:
    print("\n❌ No se encontraron cámaras disponibles")
    print("   Verifica que la cámara esté conectada y no esté en uso por otra aplicación")

