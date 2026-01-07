"""
Data preprocessing pipeline for diabetes prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from config.settings import settings
from config.logging_config import logger
from utils.exceptions import DataValidationError


class DataPreprocessor:
    """
    Handles all data preprocessing operations.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Pregnancies',
            'Glucose', 
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
        self.target_column = 'Outcome'
    
    def load_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load diabetes dataset from CSV.
        
        Args:
            filepath: Path to CSV file (uses default if None)
            
        Returns:
            DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if filepath is None:
            filepath = settings.data.raw_data_path
        
        logger.info(f"Loading data from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate dataset has required columns and reasonable values.
        """
        logger.info("Validating dataset")
        
        # Check required columns exist
        missing_cols = set(self.feature_columns + [self.target_column]) - set(df.columns)
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}",
                field=None
            )
        
        # Check for null values
        null_counts = df[self.feature_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
        
        # Check data types
        for col in self.feature_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise DataValidationError(
                    f"Column {col} must be numeric, got {df[col].dtype}",
                    field=col
                )
        
        # Check target values are binary (0 or 1)
        unique_targets = df[self.target_column].unique()
        if not set(unique_targets).issubset({0, 1}):
            raise DataValidationError(
                f"Target column must contain only 0 or 1, found: {unique_targets}",
                field=self.target_column
            )
        
        logger.info("✓ Data validation passed")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing/zero values in medical data.
        
        In medical datasets, zero often means "not measured" rather than actual zero.
        We replace with median of non-zero values.
        """
        logger.info("Handling missing values")
        
        # Columns where zero is medically impossible
        zero_not_possible = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        df = df.copy()  # Don't modify original
        
        for col in zero_not_possible:
            if col in df.columns:
                # Count zeros
                zero_count = (df[col] == 0).sum()
                
                if zero_count > 0:
                    logger.info(f"Found {zero_count} zeros in {col}, replacing with median")
                    
                    # Calculate median of non-zero values
                    non_zero_values = df[df[col] != 0][col]
                    if len(non_zero_values) > 0:
                        median_value = non_zero_values.median()
                        
                        # Replace zeros with median
                        df.loc[df[col] == 0, col] = median_value
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and report outliers using IQR method.
        
        IQR (Interquartile Range) method:
        - Q1 = 25th percentile
        - Q3 = 75th percentile
        - IQR = Q3 - Q1
        - Outliers: < Q1 - 1.5*IQR or > Q3 + 1.5*IQR
        """
        logger.info("Detecting outliers")
        
        for col in self.feature_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(outliers) > 0:
                logger.info(f"Found {len(outliers)} outliers in {col}")
                logger.debug(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Fraction of data for testing (0.2 = 20%)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Stratify ensures class balance in both train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Keep same class distribution
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Class distribution in training: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        StandardScaler: (value - mean) / std_dev
        Results in mean=0, std=1 for each feature.
        
        Args:
            X_train: Training features
            X_test: Test features
            fit: Whether to fit scaler (True for training, False for inference)
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled) as numpy arrays
        """
        logger.info("Scaling features")
        
        if fit:
            # Fit on training data
            X_train_scaled = self.scaler.fit_transform(X_train)
            logger.info(f"Scaler fitted - mean: {self.scaler.mean_[:3]}...")
            logger.info(f"Scaler fitted - std: {self.scaler.scale_[:3]}...")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        # Always just transform test data (never fit on test!)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, filepath: Optional[Path] = None) -> None:
        """
        Save fitted scaler to disk.
        """
        if filepath is None:
            filepath = settings.model.scaler_path
        
        logger.info(f"Saving scaler to {filepath}")
        
        try:
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, filepath)
            logger.info("✓ Scaler saved successfully")
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}", exc_info=True)
            raise
    
    def load_scaler(self, filepath: Optional[Path] = None) -> None:
        """
        Load fitted scaler from disk.
        """
        if filepath is None:
            filepath = settings.model.scaler_path
        
        logger.info(f"Loading scaler from {filepath}")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        try:
            self.scaler = joblib.load(filepath)
            logger.info("✓ Scaler loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}", exc_info=True)
            raise
    
    def prepare_training_data(
        self,
        filepath: Optional[Path] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for training.
        
        This is the main method that orchestrates all preprocessing steps.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) ready for model training
        """
        logger.info("="*60)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Load data
        df = self.load_data(filepath)
        
        # Step 2: Validate
        self.validate_data(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Detect outliers (just reporting, not removing)
        df = self.detect_outliers(df)
        
        # Step 5: Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # Step 7: Save scaler for later use
        self.save_scaler()
        
        logger.info("="*60)
        logger.info("✓ PREPROCESSING COMPLETE")
        logger.info("="*60)
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def prepare_single_prediction(self, features: np.ndarray) -> np.ndarray:
        """
        Prepare single patient data for prediction.
        
        Args:
            features: Array of shape (1, 8) with patient features
            
        Returns:
            Scaled features ready for model input
        """
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler not fitted. Load scaler first with load_scaler()")
        
        return self.scaler.transform(features)