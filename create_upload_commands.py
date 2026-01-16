#!/usr/bin/env python3
"""
Generate base64 upload commands for RunPod
Creates commands that can be copy-pasted into RunPod terminal
"""
import base64
import os
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
print("# Copia y pega cada bloque en RunPod")
print("# ==========================================")
print()

for local_file, remote_path in FILES_TO_UPLOAD:
    file_path = PROJECT_DIR / local_file
    if not file_path.exists():
        print(f"# ❌ Archivo no encontrado: {local_file}")
        continue
    
    print(f"# ==========================================")
    print(f"# {local_file} -> {remote_path}")
    print(f"# ==========================================")
    
    # Read file and encode to base64
    with open(file_path, 'rb') as f:
        content = f.read()
        encoded = base64.b64encode(content).decode('utf-8')
    
    # Create command for RunPod
    print(f"cat > {remote_path} << 'ENDBASE64'")
    print("import base64")
    print("exec(base64.b64decode('''")
    print(encoded)
    print("''').decode('utf-8'))")
    print("ENDBASE64")
    print()
    print("# O mejor, usar Python para decodificar:")
    print(f"python3 << 'ENDPYTHON' > {remote_path}")
    print("import base64")
    print(f"data = '''{encoded}'''")
    print("with open('" + remote_path.replace('~', '/root') + "', 'wb') as f:")
    print("    f.write(base64.b64decode(data))")
    print("ENDPYTHON")
    print()
    print()
