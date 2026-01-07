"""Verify all files are present and have correct content."""

from pathlib import Path
import sys

project_root = Path.cwd()

print("="*80)
print("VERIFYING ALL PROJECT FILES")
print("="*80)

# Required files with minimum expected size
required_files = {
    'config/__init__.py': 0,
    'config/settings.py': 1000,
    'config/logging_config.py': 1000,
    'utils/__init__.py': 0,
    'utils/exceptions.py': 500,
    'data/__init__.py': 0,
    'data/preprocessor.py': 2000,
    'models/__init__.py': 0,
    'scripts/__init__.py': 0,
}

all_good = True

for file_path, min_size in required_files.items():
    full_path = project_root / file_path
    
    if not full_path.exists():
        print(f"✗ {file_path} - MISSING")
        all_good = False
    else:
        size = full_path.stat().st_size
        if size < min_size:
            print(f"✗ {file_path} - TOO SMALL ({size} bytes, expected >{min_size})")
            all_good = False
        else:
            print(f"✓ {file_path} ({size} bytes)")

# Test imports
print("\n" + "="*80)
print("TESTING IMPORTS")
print("="*80)

sys.path.insert(0, str(project_root))

imports = [
    ('config.settings', 'settings'),
    ('config.logging_config', 'logger'),
    ('utils.exceptions', 'DataValidationError'),
    ('data.preprocessor', 'DataPreprocessor'),
]

for module_name, object_name in imports:
    try:
        module = __import__(module_name, fromlist=[object_name])
        obj = getattr(module, object_name)
        print(f"✓ from {module_name} import {object_name}")
    except ImportError as e:
        print(f"✗ from {module_name} import {object_name}")
        print(f"    Error: {e}")
        all_good = False
    except AttributeError as e:
        print(f"✗ {object_name} not found in {module_name}")
        print(f"    Error: {e}")
        all_good = False

print("\n" + "="*80)
if all_good:
    print("✓ ALL CHECKS PASSED")
    print("="*80)
    print("\nYou can now run:")
    print("  python scripts/train_model.py")
else:
    print("✗ SOME CHECKS FAILED")
    print("="*80)
    sys.exit(1)