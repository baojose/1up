#!/usr/bin/env python3
"""
Generate commands to create files on RunPod using base64.
Copy-paste the output into RunPod SSH terminal.
"""
import base64
import os
from pathlib import Path

def generate_file_command(local_path, remote_path):
    """Generate command to create file on RunPod using base64."""
    if not os.path.exists(local_path):
        print(f"# ⚠️  File not found: {local_path}")
        return None
    
    with open(local_path, 'rb') as f:
        content = f.read()
        encoded = base64.b64encode(content).decode('ascii')
    
    # Split into chunks of 1000 chars for readability
    chunk_size = 1000
    chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]
    
    print(f"# Creating {remote_path}")
    print(f"cat > {remote_path} << 'ENDOFFILE'")
    print("base64 -d << 'ENDOFBASE64' | cat")
    for chunk in chunks:
        print(chunk)
    print("ENDOFBASE64")
    print("ENDOFFILE")
    print("echo '✅ Created', {remote_path}")
    print("")
    return True

# Files to create
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
print("# Copy these commands into RunPod SSH terminal")
print("# ==========================================")
print("")

for local, remote in files:
    generate_file_command(local, remote)

print("# ==========================================")
print("# Done! Verify with: ls -la ~/1UP_2")
print("# ==========================================")
