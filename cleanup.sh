#!/bin/bash
# cleanup.sh - Clean up project

echo "🧹 Cleaning up Diabetes Risk Predictor project..."

# Remove test files
echo "Removing test files..."
rm -f test_*.py
rm -f debug_*.py
rm -f verify_*.py
rm -f diagnose.py
rm -f run_*.py

# Remove Python cache
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Remove OS files
echo "Removing OS files..."
find . -name ".DS_Store" -delete

# Remove generated files
echo "Removing generated files..."
rm -f *.png
rm -f *.log

# Remove Jupyter checkpoints
echo "Removing Jupyter checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Files remaining:"
find . -type f -name "*.py" | grep -v __pycache__ | sort