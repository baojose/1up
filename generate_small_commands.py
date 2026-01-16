#!/usr/bin/env python3
"""
Generate small commands for each file (easier to copy-paste)
"""
import base64
from pathlib import Path

PROJECT_DIR = Path("/Users/jba7790/Desktop/1UP_2")
FILES_TO_UPLOAD = [
    ("filters.py", "~/1UP_2/filters.py"),  # Start with smallest
    ("image_quality.py", "~/1UP_2/image_quality.py"),
    ("storage.py", "~/1UP_2/storage.py"),
    ("storage_v2.py", "~/1UP_2/storage_v2.py"),
    ("camera_utils.py", "~/1UP_2/camera_utils.py"),
    ("analyzer.py", "~/1UP_2/analyzer.py"),
    ("detector.py", "~/1UP_2/detector.py"),
    ("server/api.py", "~/1UP_2/server/api.py"),
    ("client/capture_client.py", "~/1UP_2/client/capture_client.py"),
]

print("# ==========================================")
print("# Comandos para ejecutar en RunPod")
print("# Ejecuta CADA bloque por separado")
print("# ==========================================")
print()

for local_file, remote_path in FILES_TO_UPLOAD:
    file_path = PROJECT_DIR / local_file
    if not file_path.exists():
        print(f"# ❌ Archivo no encontrado: {local_file}")
        continue
    
    # Read and encode
    with open(file_path, 'rb') as f:
        content = f.read()
        encoded = base64.b64encode(content).decode('utf-8')
    
    # Split into chunks
    chunk_size = 50000
    chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
    
    remote_path_abs = remote_path.replace('~', '/root')
    file_size_kb = len(content) / 1024
    
    print(f"# ==========================================")
    print(f"# {local_file} ({file_size_kb:.1f} KB)")
    print(f"# ==========================================")
    print(f"python3 << 'ENDFILE'")
    print("import base64")
    print("from pathlib import Path")
    print("")
    print("data_chunks = [")
    for chunk in chunks:
        print(f"    '''{chunk}''',")
    print("]")
    print("data = ''.join(data_chunks)")
    print(f"Path('{remote_path_abs}').parent.mkdir(parents=True, exist_ok=True)")
    print(f"with open('{remote_path_abs}', 'wb') as f:")
    print("    f.write(base64.b64decode(data))")
    print(f"print('✅ {local_file} creado')")
    print("ENDFILE")
    print()

print("# ==========================================")
print("# Verificar archivos:")
print("# ==========================================")
print("ls -la ~/1UP_2/*.py")
print("ls -la ~/1UP_2/server/*.py")
print("ls -la ~/1UP_2/client/*.py")
