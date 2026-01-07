"""
Neural network model for diabetes prediction.
Using TensorFlow/Keras.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import settings
from config.logging_config import logger
from utils.exceptions import ModelLoadError, TrainingError, PredictionError


class DiabetesNeuralNetwork:
    """
    Neural network for diabetes risk prediction.
    
    Architecture:
    - Input: 8 features (medical measurements)
    - Hidden layers: 3 layers with dropout
    - Output: 1 sigmoid unit (probability of diabetes)
    """
    
    def __init__(self):
        self.model: Optional[keras.Model] = None
        self.history: Optional[keras.callbacks.History] = None
        self.training_metrics: Dict[str, Any] = {}
    
    def build_model(
        self,
        input_dim: int = 8,
        hidden_layers: list = None,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ) -> keras.Model:
        """
        Build neural network architecture.
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons per layer (default: [64, 32, 16])
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        if hidden_layers is None:
            hidden_layers = [64, 32, 16]
        
        logger.info("="*60)
        logger.info("BUILDING NEURAL NETWORK")
        logger.info("="*60)
        logger.info(f"Input dimension: {input_dim}")
        logger.info(f"Hidden layers: {hidden_layers}")
        logger.info(f"Dropout rate: {dropout_rate}")
        logger.info(f"Learning rate: {learning_rate}")
        
        # Build model architecture
        model = keras.Sequential([
            # Input layer
            layers.Dense(
                hidden_layers[0], 
                activation='relu',
                input_dim=input_dim,
                name='input_layer'
            ),
            layers.Dropout(dropout_rate, name='dropout_1'),
            
            # Hidden layer 2
            layers.Dense(
                hidden_layers[1],
                activation='relu',
                name='hidden_layer_2'
            ),
            layers.Dropout(dropout_rate, name='dropout_2'),
            
            # Hidden layer 3
            layers.Dense(
                hidden_layers[2],
                activation='relu',
                name='hidden_layer_3'
            ),
            layers.Dropout(dropout_rate * 0.7, name='dropout_3'),
            
            # Output layer (sigmoid for binary classification)
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
        
        self.model = model
        
        # Print model summary
        logger.info("\nModel Architecture:")
        model.summary(print_fn=logger.info)
        
        # Count trainable parameters
        total_params = model.count_params()
        logger.info(f"\n✓ Model built with {total_params:,} trainable parameters")
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 200,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> keras.callbacks.History:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
            
        Returns:
            Training history object
        """
        if self.model is None:
            raise TrainingError("Model not built. Call build_model() first.")
        
        logger.info("="*60)
        logger.info("TRAINING NEURAL NETWORK")
        logger.info("="*60)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Validation samples: {len(X_val)}")
        else:
            validation_data = None
            logger.info(f"Validation split: {validation_split}")
        
        # Setup callbacks
        callback_list = self._get_callbacks()
        
        try:
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                validation_split=validation_split if validation_data is None else 0.0,
                callbacks=callback_list,
                verbose=1
            )
            
            logger.info("\n✓ Training completed successfully")
            
            # Store final metrics
            self.training_metrics = {
                'final_loss': float(self.history.history['loss'][-1]),
                'final_accuracy': float(self.history.history['accuracy'][-1]),
                'final_auc': float(self.history.history['auc'][-1])
            }
            
            if 'val_loss' in self.history.history:
                self.training_metrics.update({
                    'final_val_loss': float(self.history.history['val_loss'][-1]),
                    'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                    'final_val_auc': float(self.history.history['val_auc'][-1])
                })
            
            logger.info("\nFinal Training Metrics:")
            for metric, value in self.training_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return self.history
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise TrainingError(f"Failed to train model: {e}")
    
    def _get_callbacks(self) -> list:
        """Setup training callbacks."""
        callback_list = []
        
        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = settings.model.model_dir / 'best_model.h5'
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(model_checkpoint)
        
        logger.info(f"Configured {len(callback_list)} training callbacks")
        
        return callback_list
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise PredictionError("Model not loaded. Build or load a model first.")
        
        logger.info("="*60)
        logger.info("EVALUATING MODEL")
        logger.info("="*60)
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=['No Diabetes', 'Diabetes'],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Compile results
        results = {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'precision': float(report['Diabetes']['precision']),
            'recall': float(report['Diabetes']['recall']),
            'f1_score': float(report['Diabetes']['f1-score']),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Log results
        logger.info("\nTest Set Performance:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  AUC-ROC:   {auc:.4f}")
        logger.info(f"  Precision: {results['precision']:.4f}")
        logger.info(f"  Recall:    {results['recall']:.4f}")
        logger.info(f"  F1-Score:  {results['f1_score']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
        logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
        
        # Generate visualizations
        self._plot_confusion_matrix(cm)
        if self.history:
            self._plot_training_history()
        
        return results
    
    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (scaled)
            return_proba: Return probabilities instead of binary predictions
            
        Returns:
            Predictions (0/1) or probabilities
        """
        if self.model is None:
            raise PredictionError("Model not loaded")
        
        try:
            predictions = self.model.predict(X, verbose=0)
            
            if return_proba:
                return predictions
            else:
                return (predictions > 0.5).astype(int)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Failed to make prediction: {e}")
    
    def predict_with_confidence(
        self,
        X: np.ndarray
    ) -> Tuple[int, float, str]:
        """
        Make prediction with confidence level.
        
        Returns:
            Tuple of (prediction, probability, risk_level)
        """
        probability = self.model.predict(X, verbose=0)[0][0]
        prediction = int(probability > 0.5)
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return prediction, float(probability), risk_level
    
    def save(self, filepath: Optional[Path] = None) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            filepath = settings.model.model_path
        
        logger.info(f"Saving model to {filepath}")
        
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.save(filepath)
            logger.info("✓ Model saved successfully")
            
            # Save metadata
            metadata_path = filepath.with_suffix('.json')
            import json
            metadata = {
                'version': settings.model.version,
                'architecture': settings.model.architecture,
                'training_metrics': self.training_metrics,
                'input_features': [
                    'Pregnancies', 'Glucose', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age'
                ]
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)
            raise
    
    def load(self, filepath: Optional[Path] = None) -> None:
        """Load model from disk."""
        if filepath is None:
            filepath = settings.model.model_path
        
        logger.info(f"Loading model from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            self.model = keras.models.load_model(filepath)
            
            # Make a dummy prediction to build the model
            dummy_input = np.zeros((1, 8))
            _ = self.model.predict(dummy_input, verbose=0)
            
            logger.info("✓ Model loaded successfully")
            
            # Load metadata if exists
            metadata_path = filepath.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Model version: {metadata.get('version')}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        plot_path = settings.model.model_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Confusion matrix saved to {plot_path}")
        plt.close()
    
    def _plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0, 0].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        if 'val_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Train', linewidth=2)
        if 'val_auc' in self.history.history:
            axes[1, 0].plot(self.history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('AUC-ROC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 1].plot(self.history.history['precision'], label='Precision', linewidth=2)
        axes[1, 1].plot(self.history.history['recall'], label='Recall', linewidth=2)
        if 'val_precision' in self.history.history:
            axes[1, 1].plot(self.history.history['val_precision'], 
                          label='Val Precision', linestyle='--', linewidth=2)
        if 'val_recall' in self.history.history:
            axes[1, 1].plot(self.history.history['val_recall'], 
                          label='Val Recall', linestyle='--', linewidth=2)
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = settings.model.model_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Training history saved to {plot_path}")
        plt.close()