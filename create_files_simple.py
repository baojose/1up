#!/usr/bin/env python3
"""
Generate Python script to create files on RunPod.
Copy-paste the output into RunPod SSH terminal.
"""
import base64
import os

files = [
    ("detector.py", "~/1UP_2/detector.py"),
    ("analyzer.py", "~/1UP_2/analyzer.py"),
    ("filters.py", "~/1UP_2/filters.py"),
    ("storage.py", "~/1UP_2/storage.py"),
    ("storage_v2.py", "~/1UP_2/storage_v2.py"),
    ("image_quality.py", "~/1UP_2/image_quality.py"),
    ("camera_utils.py", "~/1UP_2/camera_utils.py"),
    ("config.yaml", "~/1UP_2/config.yaml"),
    ("requirements.txt", "~/1UP_2/requirements.txt"),
    ("server/requirements_server.txt", "~/1UP_2/server/requirements_server.txt"),
    ("server/api.py", "~/1UP_2/server/api.py"),
    ("server/config_server.yaml", "~/1UP_2/server/config_server.yaml"),
    ("client/capture_client.py", "~/1UP_2/client/capture_client.py"),
    ("client/config_client.yaml", "~/1UP_2/client/config_client.yaml"),
]

print("# ==========================================")
print("# Copy-paste this Python script into RunPod SSH terminal")
print("# ==========================================")
print("")
print("python3 << 'ENDPYTHON'")
print("import base64")
print("import os")
print("from pathlib import Path")
print("")

for local, remote in files:
    if not os.path.exists(local):
        print(f"# ⚠️  File not found: {local}")
        continue
    
    with open(local, 'rb') as f:
        content = f.read()
        encoded = base64.b64encode(content).decode('ascii')
    
    # Expand ~ to full path
    remote_expanded = remote.replace('~/', '/root/')
    remote_dir = os.path.dirname(remote_expanded)
    
    print(f"# Creating {remote}")
    print(f"Path('{remote_dir}').mkdir(parents=True, exist_ok=True)")
    print(f"with open('{remote_expanded}', 'wb') as f:")
    print(f"    f.write(base64.b64decode('''{encoded}'''))")
    print(f"print('✅ Created {remote}')")
    print("")

print("print('✅ All files created!')")
print("ENDPYTHON")
