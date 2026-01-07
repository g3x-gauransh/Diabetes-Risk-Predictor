"""
Verify project setup before training.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def verify_setup():
    """Check if everything is ready for training."""
    
    print("="*60)
    print("VERIFYING PROJECT SETUP")
    print("="*60)
    
    all_good = True
    
    # Check Python version
    print("\n1. Python Version:")
    if sys.version_info >= (3, 8):
        print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} (need 3.8+)")
        all_good = False
    
    # Check dependencies
    print("\n2. Dependencies:")
    required = ['tensorflow', 'pandas', 'numpy', 'sklearn', 'pydantic']
    for package in required:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (run: pip install {package})")
            all_good = False
    
    # Check directories
    print("\n3. Directory Structure:")
    required_dirs = ['data', 'models', 'scripts', 'config', 'utils', 'artifacts/models', 'artifacts/scalers']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"   ✓ {dir_name}/")
        else:
            print(f"   ✗ {dir_name}/ (missing)")
            all_good = False
    
    # Check data file
    print("\n4. Dataset:")
    data_path = project_root / 'data' / 'diabetes.csv'
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"   ✓ diabetes.csv ({len(df)} rows, {len(df.columns)} columns)")
        
        # Check columns
        expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        if all(col in df.columns for col in expected_cols):
            print(f"   ✓ All required columns present")
        else:
            print(f"   ✗ Missing columns")
            all_good = False
    else:
        print(f"   ✗ diabetes.csv not found")
        print(f"      Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
        print(f"      Or run: python data/download_data.py")
        all_good = False
    
    # Check configuration files
    print("\n5. Configuration:")
    config_files = ['config/settings.py', 'config/logging_config.py']
    for file_name in config_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"   ✓ {file_name}")
        else:
            print(f"   ✗ {file_name} (missing)")
            all_good = False
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✓ ALL CHECKS PASSED - READY TO TRAIN!")
        print("="*60)
        print("\nRun training with:")
        print("  python scripts/train_model.py")
    else:
        print("✗ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("="*60)
        sys.exit(1)


if __name__ == '__main__':
    verify_setup()