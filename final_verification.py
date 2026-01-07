"""Final verification before training."""

import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

print("="*80)
print("FINAL VERIFICATION")
print("="*80)

# Test all imports in order
tests = []

print("\n1. Testing config.settings...")
try:
    from config.settings import settings
    print(f"   ✓ Settings imported")
    print(f"     - Environment: {settings.environment}")
    print(f"     - Model version: {settings.model.version}")
    print(f"     - Model path: {settings.model.model_path}")
    tests.append(True)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    tests.append(False)

print("\n2. Testing config.logging_config...")
try:
    from config.logging_config import logger
    print(f"   ✓ Logger imported")
    logger.info("Test log message")
    tests.append(True)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    tests.append(False)

print("\n3. Testing utils.exceptions...")
try:
    from utils.exceptions import (
        DataValidationError,
        ModelNotFoundError,
        TrainingError,
        PredictionError
    )
    print(f"   ✓ Exceptions imported")
    tests.append(True)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    tests.append(False)

print("\n4. Testing data.preprocessor...")
try:
    from data.preprocessor import DataPreprocessor
    print(f"   ✓ DataPreprocessor imported")
    preprocessor = DataPreprocessor()
    print(f"     - Feature columns: {len(preprocessor.feature_columns)}")
    tests.append(True)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    tests.append(False)

print("\n5. Testing models.neural_network...")
try:
    from models.neural_network import DiabetesNeuralNetwork
    print(f"   ✓ DiabetesNeuralNetwork imported")
    model = DiabetesNeuralNetwork()
    print(f"     - Model initialized")
    tests.append(True)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    tests.append(False)

print("\n6. Checking data file...")
try:
    data_path = settings.data.raw_data_path
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"   ✓ Data file found: {data_path}")
        print(f"     - Rows: {len(df)}")
        print(f"     - Columns: {len(df.columns)}")
        tests.append(True)
    else:
        print(f"   ✗ Data file not found: {data_path}")
        print(f"     Run: python data/download_data.py")
        tests.append(False)
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    tests.append(False)

# Summary
print("\n" + "="*80)
passed = sum(tests)
total = len(tests)

if passed == total:
    print(f"✓ ALL {total} CHECKS PASSED - READY TO TRAIN!")
    print("="*80)
    print("\nRun training:")
    print("  python scripts/train_model.py")
    sys.exit(0)
else:
    print(f"✗ {total - passed}/{total} CHECKS FAILED")
    print("="*80)
    print("\nFix the errors above before running training.")
    sys.exit(1)