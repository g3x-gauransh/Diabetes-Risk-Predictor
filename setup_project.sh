#!/bin/bash
# setup_project.sh - Complete project setup

echo "Setting up Diabetes Risk Predictor project..."

# Create all directories
echo "Creating directories..."
mkdir -p config
mkdir -p data
mkdir -p models
mkdir -p utils
mkdir -p scripts
mkdir -p artifacts/models
mkdir -p artifacts/scalers
mkdir -p logs

# Create __init__.py files
echo "Creating __init__.py files..."
touch config/__init__.py
touch data/__init__.py
touch models/__init__.py
touch utils/__init__.py
touch scripts/__init__.py

echo "✓ Project structure created"

# Create .env file if doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
ENVIRONMENT=development
MODEL_VERSION=1.0.0
LOG_LEVEL=INFO
EPOCHS=200
BATCH_SIZE=32
LEARNING_RATE=0.001
EOF
    echo "✓ .env created"
fi

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create virtual environment: python -m venv tf_env"
echo "2. Activate it: source tf_env/bin/activate"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Download data: python data/download_data.py"
echo "5. Run training: python scripts/train_model.py"