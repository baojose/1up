#!/bin/bash
# Upload files to RunPod using base64 encoding (workaround for SCP issues)
# Usage: ./upload_base64.sh

RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="~/1UP_2"

echo "📤 Uploading files to RunPod using base64 encoding..."
echo ""

# Function to upload a file
upload_file() {
    local file=$1
    local remote_path=$2
    
    if [ ! -f "$file" ]; then
        echo "⚠️  Warning: $file not found, skipping..."
        return 1
    fi
    
    echo "📁 Uploading $file..."
    base64 "$file" | ssh -i "$SSH_KEY" "$RUNPOD_HOST" "base64 -d > $remote_path && echo '✅ $file uploaded'" || {
        echo "❌ Failed to upload $file"
        return 1
    }
}

# Upload Python files
upload_file "detector.py" "$REMOTE_DIR/detector.py"
upload_file "analyzer.py" "$REMOTE_DIR/analyzer.py"
upload_file "filters.py" "$REMOTE_DIR/filters.py"
upload_file "storage.py" "$REMOTE_DIR/storage.py"
upload_file "storage_v2.py" "$REMOTE_DIR/storage_v2.py"
upload_file "image_quality.py" "$REMOTE_DIR/image_quality.py"
upload_file "camera_utils.py" "$REMOTE_DIR/camera_utils.py"

# Upload config files
upload_file "config.yaml" "$REMOTE_DIR/config.yaml"
upload_file "requirements.txt" "$REMOTE_DIR/requirements.txt"
upload_file "server/requirements_server.txt" "$REMOTE_DIR/requirements_server.txt"

# Upload server files
upload_file "server/api.py" "$REMOTE_DIR/server/api.py"
upload_file "server/config_server.yaml" "$REMOTE_DIR/server/config_server.yaml"

# Upload client files
upload_file "client/capture_client.py" "$REMOTE_DIR/client/capture_client.py"
upload_file "client/config_client.yaml" "$REMOTE_DIR/client/config_client.yaml"

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next: SSH to RunPod and verify files:"
echo "  ssh -i $SSH_KEY $RUNPOD_HOST"
echo "  cd $REMOTE_DIR && ls -la"
