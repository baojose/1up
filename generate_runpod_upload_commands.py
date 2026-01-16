#!/usr/bin/env python3
"""
Generate Python commands to create files directly in RunPod
This avoids scp/rsync connection issues
"""
import base64
from pathlib import Path

PROJECT_DIR = Path("/Users/jba7790/Desktop/1UP_2")
FILES_TO_UPLOAD = [
    ("detector.py", "~/1UP_2/detector.py"),
    ("analyzer.py", "~/1UP_2/analyzer.py"),
    ("filters.py", "~/1UP_2/filters.py"),
    ("image_quality.py", "~/1UP_2/image_quality.py"),
    ("camera_utils.py", "~/1UP_2/camera_utils.py"),
    ("storage_v2.py", "~/1UP_2/storage_v2.py"),
    ("storage.py", "~/1UP_2/storage.py"),
    ("server/api.py", "~/1UP_2/server/api.py"),
    ("client/capture_client.py", "~/1UP_2/client/capture_client.py"),
]

print("# ==========================================")
print("# Comandos para ejecutar en RunPod")
print("# Copia y pega CADA bloque en RunPod")
print("# ==========================================")
print()

for local_file, remote_path in FILES_TO_UPLOAD:
    file_path = PROJECT_DIR / local_file
    if not file_path.exists():
        print(f"# ❌ Archivo no encontrado: {local_file}")
        continue
    
    # Read file
    with open(file_path, 'rb') as f:
        content = f.read()
        encoded = base64.b64encode(content).decode('utf-8')
    
    # Split into chunks (base64 can be very long)
    chunk_size = 10000  # Characters per chunk
    chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
    
    print(f"# ==========================================")
    print(f"# {local_file} -> {remote_path}")
    print(f"# ==========================================")
    print(f"python3 << 'ENDPYTHON'")
    print("import base64")
    print("from pathlib import Path")
    print()
    print("# Base64 data (split into chunks)")
    print("data_chunks = [")
    for i, chunk in enumerate(chunks):
        print(f"    '''{chunk}''',")
    print("]")
    print()
    print("# Combine and decode")
    print("data = ''.join(data_chunks)")
    print(f"content = base64.b64decode(data)")
    print()
    print("# Create directory if needed")
    remote_path_abs = remote_path.replace('~', '/root')
    print(f"Path('{remote_path_abs}').parent.mkdir(parents=True, exist_ok=True)")
    print()
    print("# Write file")
    print(f"with open('{remote_path_abs}', 'wb') as f:")
    print("    f.write(content)")
    print()
    print(f"print('✅ {local_file} creado')")
    print("ENDPYTHON")
    print()
    print()

print("# ==========================================")
print("# Verificar archivos creados:")
print("# ==========================================")
print("ls -la ~/1UP_2/*.py")
print("ls -la ~/1UP_2/server/*.py")
print("ls -la ~/1UP_2/client/*.py")
