#!/bin/bash
# 🚀 Upload 1UP code to RunPod - FAST METHOD
# This script uploads essential files directly to RunPod

set -e

# RunPod connection details
RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="~/1UP_2"

echo "🚀 Uploading 1UP code to RunPod..."
echo "Host: $RUNPOD_HOST"
echo ""

# Create directory structure on RunPod
echo "📁 Creating directory structure..."
ssh -i "$SSH_KEY" "$RUNPOD_HOST" "mkdir -p $REMOTE_DIR/server $REMOTE_DIR/client $REMOTE_DIR/images/crops $REMOTE_DIR/images/raw $REMOTE_DIR/database"

# Upload essential Python files
echo "📤 Uploading Python files..."
scp -i "$SSH_KEY" detector.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" analyzer.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" filters.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" image_quality.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" camera_utils.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" storage_v2.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" storage.py "$RUNPOD_HOST:$REMOTE_DIR/"
scp -i "$SSH_KEY" server/api.py "$RUNPOD_HOST:$REMOTE_DIR/server/"
scp -i "$SSH_KEY" client/capture_client.py "$RUNPOD_HOST:$REMOTE_DIR/client/"

# Upload config files
echo "📤 Uploading config files..."
scp -i "$SSH_KEY" server/config_server.yaml "$RUNPOD_HOST:$REMOTE_DIR/server/" 2>/dev/null || echo "⚠️  config_server.yaml not found locally, will create on server"
scp -i "$SSH_KEY" client/config_client.yaml "$RUNPOD_HOST:$REMOTE_DIR/client/" 2>/dev/null || echo "⚠️  config_client.yaml not found locally, will create on server"
scp -i "$SSH_KEY" server/requirements_server.txt "$RUNPOD_HOST:$REMOTE_DIR/server/"
scp -i "$SSH_KEY" requirements.txt "$RUNPOD_HOST:$REMOTE_DIR/"

echo ""
echo "✅ Upload complete!"
echo ""
echo "📝 Next steps:"
echo "1. SSH to RunPod: ssh -i $SSH_KEY $RUNPOD_HOST"
echo "2. Create config files if missing"
echo "3. Set up virtual environment"
echo "4. Install dependencies"
echo "5. Start the server"
