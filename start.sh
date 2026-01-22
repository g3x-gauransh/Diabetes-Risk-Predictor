#!/bin/bash
# start.sh - Start API and Streamlit UI with better error handling

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🏥 Starting Diabetes Risk Predictor..."
echo "======================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo "✓ API server stopped"
    fi
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Check if virtual environment exists
if [ ! -d "tf_env" ]; then
    echo "✗ Virtual environment not found"
    echo "  Create it with: python -m venv tf_env"
    exit 1
fi

# Activate virtual environment
source tf_env/bin/activate
echo "✓ Virtual environment activated"

# Create logs directory if doesn't exist
mkdir -p logs

# Check if model exists
MODEL_PATH="artifacts/models/diabetes_risk_predictor_v1.0.0.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo "✗ Model not found: $MODEL_PATH"
    echo "  Train the model first: python scripts/train_model.py"
    exit 1
fi
echo "✓ Model found"

# Check if scaler exists
SCALER_PATH="artifacts/scalers/scaler_v1.0.0.pkl"
if [ ! -f "$SCALER_PATH" ]; then
    echo "✗ Scaler not found: $SCALER_PATH"
    echo "  Train the model first: python scripts/train_model.py"
    exit 1
fi
echo "✓ Scaler found"

# Start API in background
echo ""
echo "Starting API server on port 8000..."
python api/app.py > logs/api.log 2>&1 &
API_PID=$!
echo "✓ API started (PID: $API_PID)"
echo "  Logs: tail -f logs/api.log"

# Wait for API to initialize
echo ""
echo "Waiting for API to initialize..."
for i in {1..10}; do
    sleep 1
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✓ API is ready!"
        break
    fi
    echo -n "."
    if [ $i -eq 10 ]; then
        echo ""
        echo "✗ API failed to start after 10 seconds"
        echo ""
        echo "API Log Output:"
        echo "==============="
        cat logs/api.log
        echo "==============="
        exit 1
    fi
done

# Check API health
echo ""
echo "Checking API health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
echo "Health check response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "✓ API health check passed"
else
    echo "✗ API health check failed"
    echo ""
    echo "API Log Output:"
    cat logs/api.log
    exit 1
fi

# Start Streamlit UI
echo ""
echo "======================================"
echo "Starting Streamlit UI..."
echo "======================================"
echo ""
echo "🌐 Access points:"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • Web Interface: http://localhost:8501"
echo "  • API Logs: tail -f logs/api.log"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "======================================"
echo ""

# Start Streamlit (runs in foreground)
streamlit run ui/streamlit_app.py --server.port 8501 --server.address localhost

# Cleanup happens automatically via trap