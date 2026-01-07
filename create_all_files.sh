#!/bin/bash
# create_all_files.sh - Ensure all __init__.py files exist

echo "Creating all __init__.py files..."

# Create empty __init__.py in all package directories
touch config/__init__.py
touch data/__init__.py
touch models/__init__.py
touch utils/__init__.py
touch scripts/__init__.py
touch api/__init__.py
touch services/__init__.py
touch tests/__init__.py

echo "✓ All __init__.py files created"

# List all created files
echo ""
echo "Verification:"
ls -la */__init__.py

echo ""
echo "✓ Setup complete!"