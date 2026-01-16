#!/bin/bash
# Create essential files on RunPod using cat with heredoc
# Usage: Run this script and copy-paste the output commands into RunPod SSH terminal

RUNPOD_HOST="ytoissxrquxq5s-6441116d@ssh.runpod.io"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "📝 Copy these commands and paste them into your RunPod SSH terminal:"
echo ""
echo "# =========================================="
echo "# Step 1: Create detector.py"
echo "# =========================================="
echo ""

# Read detector.py and create heredoc command
cat << 'EOF'
cat > ~/1UP_2/detector.py << 'DETECTOR_EOF'
EOF
cat detector.py
cat << 'EOF'
DETECTOR_EOF
EOF

echo ""
echo "# =========================================="
echo "# Step 2: Create server/api.py"
echo "# =========================================="
echo ""

cat << 'EOF'
cat > ~/1UP_2/server/api.py << 'API_EOF'
EOF
cat server/api.py
cat << 'EOF'
API_EOF
EOF

echo ""
echo "# =========================================="
echo "# Step 3: Create server/config_server.yaml"
echo "# =========================================="
echo ""

cat << 'EOF'
cat > ~/1UP_2/server/config_server.yaml << 'CONFIG_EOF'
EOF
cat server/config_server.yaml
cat << 'EOF'
CONFIG_EOF
EOF

echo ""
echo "✅ Script complete! Copy-paste the commands above into RunPod SSH terminal."
