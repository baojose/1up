#!/bin/bash
# Upload code to RunPod - Alternative to rsync
# Usage: ./upload_to_runpod.sh

RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="~/1UP_2"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

echo "📤 Uploading code to RunPod..."
echo ""

# Verify SSH key exists
if [ ! -f "$SSH_KEY" ]; then
    echo "❌ Error: SSH key not found at $SSH_KEY"
    exit 1
fi

# Create remote directory
echo "📁 Creating remote directory..."
ssh $SSH_OPTS $RUNPOD_HOST "mkdir -p $REMOTE_DIR" || {
    echo "❌ Error: Cannot connect to RunPod. Is the pod running?"
    exit 1
}

# Upload essential files and directories
echo "📁 Uploading Python files..."
scp $SSH_OPTS detector.py analyzer.py filters.py storage.py storage_v2.py image_quality.py camera_utils.py $RUNPOD_HOST:$REMOTE_DIR/ || {
    echo "⚠️  Warning: Some Python files failed to upload"
}

echo "📁 Uploading server directory..."
scp $SSH_OPTS -r server/ $RUNPOD_HOST:$REMOTE_DIR/ || {
    echo "⚠️  Warning: Server directory failed to upload"
}

echo "📁 Uploading client directory..."
scp $SSH_OPTS -r client/ $RUNPOD_HOST:$REMOTE_DIR/ || {
    echo "⚠️  Warning: Client directory failed to upload"
}

echo "📁 Uploading config..."
scp $SSH_OPTS config.yaml $RUNPOD_HOST:$REMOTE_DIR/ || {
    echo "⚠️  Warning: Config file failed to upload"
}

echo "📁 Uploading requirements..."
scp $SSH_OPTS requirements.txt server/requirements_server.txt $RUNPOD_HOST:$REMOTE_DIR/ || {
    echo "⚠️  Warning: Requirements files failed to upload"
}

echo "📁 Uploading SAM3 (excluding large files)..."
ssh $SSH_OPTS $RUNPOD_HOST "mkdir -p $REMOTE_DIR/sam3" || true
scp $SSH_OPTS -r sam3/sam3/ $RUNPOD_HOST:$REMOTE_DIR/sam3/ || {
    echo "⚠️  Warning: SAM3 directory failed to upload"
}
scp $SSH_OPTS sam3/pyproject.toml sam3/MANIFEST.in $RUNPOD_HOST:$REMOTE_DIR/sam3/ 2>/dev/null || true

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps (SSH to RunPod):"
echo "  ssh -i $SSH_KEY $RUNPOD_HOST"
echo "  cd $REMOTE_DIR"
echo "  # Follow docs/RUNPOD_SETUP.md"
