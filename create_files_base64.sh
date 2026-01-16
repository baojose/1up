#!/bin/bash
# Create files on RunPod using base64 encoding
# Usage: ./create_files_base64.sh

RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "📤 Creating files on RunPod using base64..."

# Function to create a file
create_file() {
    local file=$1
    local remote_path=$2
    
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: $file not found, skipping..."
        return 1
    fi
    
    echo "📁 Creating $remote_path..."
    base64 -i "$file" | ssh -i "$SSH_KEY" "$RUNPOD_HOST" "base64 -d > $remote_path && echo '✅ Created $remote_path'" || {
        echo "❌ Failed to create $remote_path"
        return 1
    }
}

# Create essential Python files
create_file "detector.py" "~/1UP_2/detector.py"
create_file "analyzer.py" "~/1UP_2/analyzer.py"
create_file "filters.py" "~/1UP_2/filters.py"
create_file "storage.py" "~/1UP_2/storage.py"
create_file "storage_v2.py" "~/1UP_2/storage_v2.py"
create_file "image_quality.py" "~/1UP_2/image_quality.py"
create_file "camera_utils.py" "~/1UP_2/camera_utils.py"

# Create config files
create_file "config.yaml" "~/1UP_2/config.yaml"
create_file "requirements.txt" "~/1UP_2/requirements.txt"
create_file "server/requirements_server.txt" "~/1UP_2/server/requirements_server.txt"

# Create server files
create_file "server/api.py" "~/1UP_2/server/api.py"
create_file "server/config_server.yaml" "~/1UP_2/server/config_server.yaml"

# Create client files
create_file "client/capture_client.py" "~/1UP_2/client/capture_client.py"
create_file "client/config_client.yaml" "~/1UP_2/client/config_client.yaml"

echo ""
echo "✅ Done!"
