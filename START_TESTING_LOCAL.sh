#!/bin/bash
# Quick start script for local testing
# Usage: ./START_TESTING_LOCAL.sh

echo "🚀 Starting Local Testing Setup"
echo "================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "   Activating venv..."
    source venv/bin/activate
fi

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server is already running"
else
    echo "📡 Starting server..."
    echo "   Open a new terminal and run:"
    echo "   cd $(pwd)/server && python api.py"
    echo ""
    echo "   Then press ENTER here to continue..."
    read
fi

# Test health endpoint
echo ""
echo "🔍 Testing health endpoint..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ Server is healthy"
else
    echo "❌ Server is not responding"
    exit 1
fi

# Run test script
echo ""
echo "🧪 Running test script..."
python test_server_local.py

echo ""
echo "✅ Testing complete!"
echo ""
echo "Next steps:"
echo "  1. If tests passed, try the full client:"
echo "     python client/capture_client.py"
echo ""
echo "  2. If everything works, you're ready for RunPod!"
