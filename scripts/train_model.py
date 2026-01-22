"""
Complete training pipeline for diabetes risk prediction model.

This script:
1. Loads and preprocesses data
2. Builds neural network
3. Trains with validation
4. Evaluates performance
5. Validates predictions make sense
6. Saves model and artifacts

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 150 --batch-size 64
"""

print("=" * 80)
print("SCRIPT STARTED - Loading modules...")
print("=" * 80)
import sys

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Then continue with rest of imports...
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Any

print("✓ Basic imports successful")

# Add project root to path
project_root = Path(__file__).parent.parent
print(f"Project root: {project_root}")
sys.path.insert(0, str(project_root))

print("Importing project modules...")

import sys
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import logger
from data.preprocessor import DataPreprocessor
from models.neural_network import DiabetesNeuralNetwork
from utils.exceptions import TrainingError


class ModelTrainer:
    """
    Orchestrates the complete model training pipeline.

    Similar to:
    - MLflow training runs
    - Kubeflow pipelines
    - SageMaker training jobs
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize trainer with configuration.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.preprocessor = DataPreprocessor()
        self.model = DiabetesNeuralNetwork()
        self.results = {}

    def validate_model_predictions(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> bool:
        """
        Validate that model predictions make sense.

        This is a sanity check to ensure the model learned correctly.
        Similar to unit tests but for model behavior.

        Args:
            X_test: Test features (scaled)
            y_test: Test labels

        Returns:
            True if validation passes, False otherwise
        """
        logger.info("=" * 60)
        logger.info("VALIDATING MODEL PREDICTIONS")
        logger.info("=" * 60)

        # Test cases with expected behavior
        test_cases = [
            {
                "name": "Low risk patient",
                "features": [1, 85, 66, 29, 0, 26.6, 0.351, 31],  # Young, healthy
                "expected_range": (0.0, 0.4),
                "description": "Young with normal glucose",
            },
            {
                "name": "Medium risk patient",
                "features": [5, 120, 80, 30, 100, 30.0, 0.5, 45],  # Middle-aged
                "expected_range": (0.3, 0.7),
                "description": "Middle-aged with borderline values",
            },
            {
                "name": "High risk patient",
                "features": [8, 180, 90, 40, 150, 40.0, 0.9, 60],  # High risk factors
                "expected_range": (0.6, 1.0),
                "description": "Multiple high-risk factors",
            },
        ]

        all_passed = True

        for test_case in test_cases:
            # Scale features
            features = np.array([test_case["features"]])
            scaled_features = self.preprocessor.scaler.transform(features)

            # Get prediction
            probability = self.model.predict(scaled_features, return_proba=True)[0][0]

            # Check if in expected range
            min_expected, max_expected = test_case["expected_range"]
            passed = min_expected <= probability <= max_expected

            status = "✓ PASS" if passed else "✗ FAIL"

            logger.info(f"\n{test_case['name']}:")
            logger.info(f"  Description: {test_case['description']}")
            logger.info(f"  Predicted probability: {probability:.4f}")
            logger.info(f"  Expected range: {min_expected:.2f} - {max_expected:.2f}")
            logger.info(f"  Status: {status}")

            if not passed:
                all_passed = False
                logger.error(f"  ⚠ Validation failed for {test_case['name']}")

        # Additional validation: Check test set performance
        logger.info("\nTest Set Validation:")

        # Get predictions on test set
        y_pred_proba = self.model.predict(X_test, return_proba=True)

        # Check predictions are in valid range
        if np.any(y_pred_proba < 0) or np.any(y_pred_proba > 1):
            logger.error("  ✗ Predictions outside valid range [0, 1]")
            all_passed = False
        else:
            logger.info("  ✓ All predictions in valid range [0, 1]")

        # Check reasonable distribution
        low_risk = (y_pred_proba < 0.3).sum()
        medium_risk = ((y_pred_proba >= 0.3) & (y_pred_proba < 0.7)).sum()
        high_risk = (y_pred_proba >= 0.7).sum()

        logger.info(f"  Prediction distribution:")
        logger.info(
            f"    Low risk (<30%):     {low_risk} ({low_risk/len(y_pred_proba)*100:.1f}%)"
        )
        logger.info(
            f"    Medium risk (30-70%): {medium_risk} ({medium_risk/len(y_pred_proba)*100:.1f}%)"
        )
        logger.info(
            f"    High risk (>70%):    {high_risk} ({high_risk/len(y_pred_proba)*100:.1f}%)"
        )

        logger.info("\n" + "=" * 60)
        if all_passed:
            logger.info("✓ ALL VALIDATION CHECKS PASSED")
        else:
            logger.error("✗ VALIDATION FAILED - MODEL MAY NEED RETRAINING")
        logger.info("=" * 60 + "\n")

        return all_passed

    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Execute complete training pipeline.

        This is the main orchestration method that runs all steps.

        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info("=" * 80)
            logger.info("DIABETES RISK PREDICTION MODEL - TRAINING PIPELINE")
            logger.info("=" * 80)
            logger.info(f"Model version: {settings.model.version}")
            logger.info(f"Environment: {settings.environment}")
            logger.info("=" * 80 + "\n")

            # ===================================================================
            # STEP 1: DATA PREPROCESSING
            # ===================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: DATA PREPROCESSING")
            logger.info("=" * 80)

            X_train, X_test, y_train, y_test = self.preprocessor.prepare_training_data(
                filepath=self.config.get("data_path")
            )

            logger.info(f"\n✓ Data preprocessing complete")
            logger.info(f"  Training samples: {len(X_train)}")
            logger.info(f"  Test samples: {len(X_test)}")
            logger.info(f"  Features: {X_train.shape[1]}")
            logger.info(
                f"  Class distribution (train): "
                f"No diabetes: {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%), "
                f"Diabetes: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)"
            )

            self.results["data"] = {
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": X_train.shape[1],
                "class_balance": {
                    "train_negative": int((y_train == 0).sum()),
                    "train_positive": int((y_train == 1).sum()),
                    "test_negative": int((y_test == 0).sum()),
                    "test_positive": int((y_test == 1).sum()),
                },
            }

            # ===================================================================
            # STEP 2: BUILD MODEL
            # ===================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: MODEL ARCHITECTURE")
            logger.info("=" * 80)

            self.model.build_model(
                input_dim=X_train.shape[1],
                hidden_layers=self.config.get(
                    "hidden_layers", settings.model.hidden_layers
                ),
                dropout_rate=self.config.get(
                    "dropout_rate", settings.model.dropout_rate
                ),
                learning_rate=self.config.get(
                    "learning_rate", settings.model.learning_rate
                ),
            )

            # ===================================================================
            # STEP 3: TRAIN MODEL
            # ===================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: MODEL TRAINING")
            logger.info("=" * 80)

            history = self.model.train(
                X_train,
                y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=self.config.get("epochs", settings.model.epochs),
                batch_size=self.config.get("batch_size", settings.model.batch_size),
            )

            self.results["training"] = self.model.training_metrics

            # ===================================================================
            # STEP 4: EVALUATE MODEL
            # ===================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: MODEL EVALUATION")
            logger.info("=" * 80)

            evaluation_results = self.model.evaluate(
                X_test, y_test, threshold=self.config.get("threshold", 0.5)
            )

            self.results["evaluation"] = evaluation_results

            # ===================================================================
            # STEP 5: VALIDATE PREDICTIONS
            # ===================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: PREDICTION VALIDATION")
            logger.info("=" * 80)

            validation_passed = self.validate_model_predictions(X_test, y_test)

            if not validation_passed:
                logger.warning("\n⚠ WARNING: Model validation failed!")
                logger.warning("The model may not have learned correctly.")
                logger.warning("Consider:")
                logger.warning("  - Retraining with different hyperparameters")
                logger.warning("  - Increasing training data")
                logger.warning("  - Checking data quality")

                if not self.config.get("force_save", False):
                    raise TrainingError("Model validation failed. Not saving model.")

            self.results["validation_passed"] = validation_passed

            # ===================================================================
            # STEP 6: SAVE MODEL AND ARTIFACTS
            # ===================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6: SAVING MODEL AND ARTIFACTS")
            logger.info("=" * 80)

            # Save model
            self.model.save()

            # Save training results
            self._save_training_report()

            # ===================================================================
            # STEP 7: FINAL SUMMARY
            # ===================================================================
            self._print_training_summary()

            return self.results

        except Exception as e:
            logger.error(f"\n✗ Training pipeline failed: {e}", exc_info=True)
            raise TrainingError(f"Training pipeline failed: {e}")

    def _save_training_report(self) -> None:
        """
        Save comprehensive training report to file.

        Creates a JSON file with all training details.
        """
        import json
        from datetime import datetime

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": settings.model.version,
            "environment": settings.environment,
            "configuration": {
                "epochs": self.config.get("epochs", settings.model.epochs),
                "batch_size": self.config.get("batch_size", settings.model.batch_size),
                "learning_rate": self.config.get(
                    "learning_rate", settings.model.learning_rate
                ),
                "hidden_layers": self.config.get(
                    "hidden_layers", settings.model.hidden_layers
                ),
                "dropout_rate": self.config.get(
                    "dropout_rate", settings.model.dropout_rate
                ),
            },
            "results": {
                "data": self.results.get("data", {}),
                "training_metrics": self.results.get("training", {}),
                "evaluation_metrics": {
                    "accuracy": self.results["evaluation"]["accuracy"],
                    "auc": self.results["evaluation"]["auc"],
                    "precision": self.results["evaluation"]["precision"],
                    "recall": self.results["evaluation"]["recall"],
                    "f1_score": self.results["evaluation"]["f1_score"],
                },
                "validation_passed": self.results.get("validation_passed", False),
            },
        }

        # Save report
        report_path = (
            settings.model.model_dir / f"training_report_{settings.model.version}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"✓ Training report saved to {report_path}")

    def _print_training_summary(self) -> None:
        """
        Print comprehensive training summary.
        """
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE - SUMMARY")
        logger.info("=" * 80)

        # Model information
        logger.info("\n📊 MODEL INFORMATION:")
        logger.info(f"  Version: {settings.model.version}")
        logger.info(f"  Architecture: {settings.model.architecture}")
        logger.info(f"  Total Parameters: {self.model.model.count_params():,}")

        # Data information
        logger.info("\n📁 DATA:")
        data_info = self.results.get("data", {})
        logger.info(f"  Training samples: {data_info.get('train_samples', 0)}")
        logger.info(f"  Test samples: {data_info.get('test_samples', 0)}")
        logger.info(f"  Features: {data_info.get('n_features', 0)}")

        # Training metrics
        logger.info("\n🎯 TRAINING METRICS:")
        training = self.results.get("training", {})
        if training:
            logger.info(f"  Final Training Loss: {training.get('final_loss', 0):.4f}")
            logger.info(
                f"  Final Training Accuracy: {training.get('final_accuracy', 0):.4f}"
            )
            logger.info(f"  Final Training AUC: {training.get('final_auc', 0):.4f}")

            if "final_val_loss" in training:
                logger.info(
                    f"  Final Validation Loss: {training.get('final_val_loss', 0):.4f}"
                )
                logger.info(
                    f"  Final Validation Accuracy: {training.get('final_val_accuracy', 0):.4f}"
                )
                logger.info(
                    f"  Final Validation AUC: {training.get('final_val_auc', 0):.4f}"
                )

        # Evaluation metrics
        logger.info("\n📈 TEST SET PERFORMANCE:")
        evaluation = self.results.get("evaluation", {})
        logger.info(
            f"  Accuracy:  {evaluation.get('accuracy', 0):.4f} ({evaluation.get('accuracy', 0)*100:.2f}%)"
        )
        logger.info(f"  AUC-ROC:   {evaluation.get('auc', 0):.4f}")
        logger.info(f"  Precision: {evaluation.get('precision', 0):.4f}")
        logger.info(f"  Recall:    {evaluation.get('recall', 0):.4f}")
        logger.info(f"  F1-Score:  {evaluation.get('f1_score', 0):.4f}")

        # Confusion matrix
        if "confusion_matrix" in evaluation:
            cm = evaluation["confusion_matrix"]
            logger.info(f"\n  Confusion Matrix:")
            logger.info(f"    True Negatives:  {cm[0][0]}")
            logger.info(f"    False Positives: {cm[0][1]}")
            logger.info(f"    False Negatives: {cm[1][0]}")
            logger.info(f"    True Positives:  {cm[1][1]}")

        # Validation status
        logger.info("\n✅ VALIDATION:")
        if self.results.get("validation_passed", False):
            logger.info("  Status: PASSED ✓")
        else:
            logger.info("  Status: FAILED ✗")

        # Saved artifacts
        logger.info("\n💾 SAVED ARTIFACTS:")
        logger.info(f"  Model: {settings.model.model_path}")
        logger.info(f"  Scaler: {settings.model.scaler_path}")
        logger.info(
            f"  Confusion Matrix: {settings.model.model_dir / 'confusion_matrix.png'}"
        )
        logger.info(
            f"  Training History: {settings.model.model_dir / 'training_history.png'}"
        )
        logger.info(
            f"  Training Report: {settings.model.model_dir / f'training_report_{settings.model.version}.json'}"
        )

        logger.info("\n" + "=" * 80)
        logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("\nNext steps:")
        logger.info("  1. Review training_history.png for learning curves")
        logger.info("  2. Check confusion_matrix.png for prediction patterns")
        logger.info("  3. Run 'python scripts/evaluate_model.py' for detailed analysis")
        logger.info("  4. Start API server: 'python api/app.py'")
        logger.info("=" * 80 + "\n")


def parse_arguments():
    """
    Parse command line arguments.

    Similar to Java's Apache Commons CLI or JCommander.
    """
    parser = argparse.ArgumentParser(
        description="Train Diabetes Risk Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size for training"
    )

    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=None,
        help="Dropout rate for regularization",
    )

    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes (e.g., --hidden-layers 64 32 16)",
    )

    # Data parameters
    parser.add_argument(
        "--data-path", type=Path, default=None, help="Path to training data CSV file"
    )

    # Other options
    parser.add_argument(
        "--force-save", action="store_true", help="Save model even if validation fails"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def main():
    """
    Main entry point for training script.

    This is what gets executed when you run:
        python scripts/train_model.py
    """
    # Parse command line arguments
    args = parse_arguments()

    # Update logging level if specified
    if args.log_level:
        logger.setLevel(args.log_level)

    # Build configuration from arguments
    config = {}

    if args.epochs is not None:
        config["epochs"] = args.epochs

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate

    if args.dropout_rate is not None:
        config["dropout_rate"] = args.dropout_rate

    if args.hidden_layers is not None:
        config["hidden_layers"] = args.hidden_layers

    if args.data_path is not None:
        config["data_path"] = args.data_path

    if args.force_save:
        config["force_save"] = True

    # Log configuration
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    # Create trainer and run pipeline
    try:
        trainer = ModelTrainer(config=config)
        results = trainer.run_training_pipeline()

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n\nTraining interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\nTraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
