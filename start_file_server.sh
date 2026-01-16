#!/bin/bash
# Start HTTP server to serve files to RunPod
# Usage: ./start_file_server.sh

PORT=8888
echo "🌐 Starting HTTP server on port $PORT..."
echo "📁 Serving files from: $(pwd)"
echo ""
echo "⚠️  Keep this terminal open!"
echo "📋 In RunPod, run the download commands (will be provided next)"
echo ""
python3 -m http.server $PORT
