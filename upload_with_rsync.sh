#!/bin/bash
# Upload files to RunPod using rsync (more robust than scp)

RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="~/1UP_2"
PROJECT_DIR="/Users/jba7790/Desktop/1UP_2"

cd "$PROJECT_DIR"

echo "🚀 Subiendo archivos con rsync..."
echo ""

# Upload Python files to root
echo "📤 Subiendo archivos Python principales..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
  detector.py analyzer.py filters.py image_quality.py camera_utils.py storage_v2.py storage.py \
  "$RUNPOD_HOST:$REMOTE_DIR/"

# Upload server/api.py
echo ""
echo "📤 Subiendo server/api.py..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
  server/api.py \
  "$RUNPOD_HOST:$REMOTE_DIR/server/"

# Upload client/capture_client.py
echo ""
echo "📤 Subiendo client/capture_client.py..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
  client/capture_client.py \
  "$RUNPOD_HOST:$REMOTE_DIR/client/"

echo ""
echo "✅ Archivos subidos!"
echo ""
echo "Verifica en RunPod con:"
echo "  ls -la ~/1UP_2/*.py"
echo "  ls -la ~/1UP_2/server/*.py"
echo "  ls -la ~/1UP_2/client/*.py"
