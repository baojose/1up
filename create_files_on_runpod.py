#!/usr/bin/env python3
"""
Generate a single Python script that creates all files on RunPod
Uses base64 encoding but splits into manageable chunks
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

output = []
output.append("#!/usr/bin/env python3")
output.append("# Script para crear archivos en RunPod")
output.append("import base64")
output.append("from pathlib import Path")
output.append("")

for local_file, remote_path in FILES_TO_UPLOAD:
    file_path = PROJECT_DIR / local_file
    if not file_path.exists():
        print(f"❌ Archivo no encontrado: {local_file}")
        continue
    
    # Read and encode
    with open(file_path, 'rb') as f:
        content = f.read()
        encoded = base64.b64encode(content).decode('utf-8')
    
    # Split into chunks of 50000 chars (safe for terminal)
    chunk_size = 50000
    chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
    
    remote_path_abs = remote_path.replace('~', '/root')
    
    output.append(f"# {local_file}")
    output.append(f"print('📤 Creando {local_file}...')")
    output.append("data_chunks = [")
    for chunk in chunks:
        output.append(f"    '''{chunk}''',")
    output.append("]")
    output.append("data = ''.join(data_chunks)")
    output.append(f"Path('{remote_path_abs}').parent.mkdir(parents=True, exist_ok=True)")
    output.append(f"with open('{remote_path_abs}', 'wb') as f:")
    output.append("    f.write(base64.b64decode(data))")
    output.append(f"print('✅ {local_file} creado')")
    output.append("")

output.append("print('')")
output.append("print('✅ Todos los archivos creados!')")
output.append("print('Verifica con: ls -la ~/1UP_2/*.py')")

# Write to file
output_file = PROJECT_DIR / "create_files_runpod.py"
with open(output_file, 'w') as f:
    f.write('\n'.join(output))

print(f"✅ Script generado: {output_file}")
print(f"📏 Tamaño del script: {output_file.stat().st_size / 1024:.1f} KB")
print("")
print("📋 Para usar en RunPod:")
print(f"   1. Copia el contenido de {output_file}")
print("   2. Pega en RunPod y ejecuta: python3 < contenido_pegado")
print("")
print("O mejor, ejecuta esto en tu Mac para mostrar el contenido:")
