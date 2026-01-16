#!/bin/bash
# Generate commands to create Python files on RunPod
# This script generates base64-encoded commands

FILES=(
    "filters.py"
    "image_quality.py"
    "camera_utils.py"
    "analyzer.py"
    "storage_v2.py"
    "storage.py"
    "detector.py"
    "server/api.py"
    "client/capture_client.py"
)

echo "# =========================================="
echo "# Copy-paste this into RunPod SSH terminal"
echo "# =========================================="
echo ""
echo "python3 << 'ENDPYTHON'"
echo "import base64"
echo "import os"
echo "from pathlib import Path"
echo ""

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "# ⚠️  File not found: $file"
        continue
    fi
    
    # Get remote path
    if [[ "$file" == server/* ]]; then
        remote_path="/root/1UP_2/$file"
        remote_dir="/root/1UP_2/server"
    elif [[ "$file" == client/* ]]; then
        remote_path="/root/1UP_2/$file"
        remote_dir="/root/1UP_2/client"
    else
        remote_path="/root/1UP_2/$file"
        remote_dir="/root/1UP_2"
    fi
    
    # Encode file
    encoded=$(base64 -i "$file" 2>/dev/null || base64 "$file" 2>/dev/null)
    
    echo "# Creating $file"
    echo "Path('$remote_dir').mkdir(parents=True, exist_ok=True)"
    echo "with open('$remote_path', 'wb') as f:"
    echo "    f.write(base64.b64decode('''$encoded'''))"
    echo "print('✅ Created $file')"
    echo ""
done

echo "print('✅ All Python files created!')"
echo "ENDPYTHON"
