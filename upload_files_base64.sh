#!/bin/bash
# Upload files to RunPod using base64 encoding
# This avoids scp connection issues

RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="~/1UP_2"
PROJECT_DIR="/Users/jba7790/Desktop/1UP_2"

cd "$PROJECT_DIR"

upload_file_base64() {
    local file=$1
    local dest=$2
    local filename=$(basename "$file")
    
    echo "📤 Subiendo $filename..."
    
    # Encode file to base64 (macOS syntax) and upload via SSH
    base64 -i "$file" | ssh -i "$SSH_KEY" -o ConnectTimeout=10 "$RUNPOD_HOST" "base64 -d > $dest && echo '✅ $filename subido' || echo '❌ Error: $filename'"
}

# Upload files one by one
upload_file_base64 "detector.py" "$REMOTE_DIR/detector.py"
upload_file_base64 "analyzer.py" "$REMOTE_DIR/analyzer.py"
upload_file_base64 "filters.py" "$REMOTE_DIR/filters.py"
upload_file_base64 "image_quality.py" "$REMOTE_DIR/image_quality.py"
upload_file_base64 "camera_utils.py" "$REMOTE_DIR/camera_utils.py"
upload_file_base64 "storage_v2.py" "$REMOTE_DIR/storage_v2.py"
upload_file_base64 "storage.py" "$REMOTE_DIR/storage.py"
upload_file_base64 "server/api.py" "$REMOTE_DIR/server/api.py"
upload_file_base64 "client/capture_client.py" "$REMOTE_DIR/client/capture_client.py"

echo ""
echo "✅ Proceso completado!"
echo ""
echo "Verifica en RunPod con:"
echo "  ls -la ~/1UP_2/*.py"
echo "  ls -la ~/1UP_2/server/*.py"
echo "  ls -la ~/1UP_2/client/*.py"
