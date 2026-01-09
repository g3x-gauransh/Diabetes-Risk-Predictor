"""
Prediction service - Business logic for diabetes risk predictions.

Separates business logic from API layer.
Similar to Spring's @Service layer in Java.
"""
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

from config.settings import settings
from config.logging_config import logger
from models.neural_network import DiabetesNeuralNetwork
from data.preprocessor import DataPreprocessor
from utils.validators import PatientData, PredictionResponse
from utils.exceptions import PredictionError, ModelNotFoundError


class PredictionService:
    """
    Service layer for making diabetes predictions.
    
    Similar to Java's @Service classes in Spring Boot.
    Handles business logic and model interaction.
    """
    
    def __init__(self):
        """Initialize service with model and preprocessor."""
        self.model = None
        self.preprocessor = None
        self._load_model_and_scaler()
    
    def _load_model_and_scaler(self) -> None:
        """
        Load trained model and scaler on service initialization.
        
        This is called once when the application starts.
        Similar to @PostConstruct in Spring.
        """
        logger.info("Initializing Prediction Service...")
        
        try:
            # Check if model exists
            if not settings.model.model_path.exists():
                raise ModelNotFoundError(str(settings.model.model_path))
            
            if not settings.model.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {settings.model.scaler_path}")
            
            # Load model
            logger.info("Loading neural network model...")
            self.model = DiabetesNeuralNetwork()
            self.model.load(settings.model.model_path)
            
            # Load scaler
            logger.info("Loading feature scaler...")
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_scaler(settings.model.scaler_path)
            
            logger.info("✓ Prediction service initialized successfully")
            logger.info(f"  Model version: {settings.model.version}")
            logger.info(f"  Model path: {settings.model.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction service: {e}", exc_info=True)
            raise
    
    def predict_single(self, patient_data: PatientData) -> PredictionResponse:
        """
        Make prediction for a single patient.
        
        Args:
            patient_data: Validated patient data
            
        Returns:
            Prediction response with risk assessment
            
        Raises:
            PredictionError: If prediction fails
        """
        logger.info("Making single prediction", extra={
            'patient_age': patient_data.age,
            'patient_glucose': patient_data.glucose
        })
        
        try:
            # Convert patient data to array
            features = patient_data.to_array()
            
            # Scale features using the loaded scaler
            features_scaled = self.preprocessor.prepare_single_prediction(features)
            
            # Make prediction
            prediction, probability, risk_level = self.model.predict_with_confidence(
                features_scaled
            )
            
            # Calculate confidence (distance from decision boundary)
            # Closer to 0 or 1 = higher confidence
            confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                patient_data, prediction, probability
            )
            
            # Create response
            response = PredictionResponse(
                prediction=prediction,
                probability=probability,
                risk_level=risk_level,
                confidence=confidence,
                recommendations=recommendations
            )
            
            logger.info(
                f"Prediction completed: {risk_level} risk ({probability:.2%})",
                extra={
                    'prediction': prediction,
                    'probability': probability,
                    'risk_level': risk_level
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Failed to make prediction: {e}")
    
    def predict_batch(self, patients: List[PatientData]) -> List[PredictionResponse]:
        """
        Make predictions for multiple patients.
        
        More efficient than calling predict_single multiple times.
        
        Args:
            patients: List of validated patient data
            
        Returns:
            List of prediction responses
        """
        logger.info(f"Making batch prediction for {len(patients)} patients")
        
        try:
            # Convert all patients to array
            features_list = [patient.to_array() for patient in patients]
            features = np.vstack(features_list)  # Stack into single array
            
            # Scale all features at once
            features_scaled = self.preprocessor.scaler.transform(features)
            
            # Make predictions for all patients at once
            probabilities = self.model.predict(features_scaled, return_proba=True).flatten()
            
            # Create responses
            responses = []
            for i, (patient, probability) in enumerate(zip(patients, probabilities)):
                prediction = int(probability > 0.5)
                
                if probability < 0.3:
                    risk_level = "Low"
                elif probability < 0.7:
                    risk_level = "Medium"
                else:
                    risk_level = "High"
                
                confidence = abs(probability - 0.5) * 2
                recommendations = self._generate_recommendations(patient, prediction, probability)
                
                response = PredictionResponse(
                    prediction=prediction,
                    probability=float(probability),
                    risk_level=risk_level,
                    confidence=confidence,
                    recommendations=recommendations
                )
                
                responses.append(response)
            
            logger.info(f"✓ Batch prediction completed for {len(patients)} patients")
            
            return responses
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise PredictionError(f"Failed to make batch prediction: {e}")
    
    def _generate_recommendations(
        self, 
        patient: PatientData, 
        prediction: int, 
        probability: float
    ) -> List[str]:
        """
        Generate personalized health recommendations.
        
        Args:
            patient: Patient data
            prediction: Binary prediction (0 or 1)
            probability: Probability of diabetes
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # High risk recommendations
        if probability > 0.7:
            recommendations.append("Consult with healthcare provider immediately")
            recommendations.append("Schedule comprehensive diabetes screening")
            recommendations.append("Monitor blood glucose levels daily")
        elif probability > 0.5:
            recommendations.append("Consult with healthcare provider")
            recommendations.append("Consider diabetes screening test")
            recommendations.append("Monitor blood glucose levels regularly")
        
        # Glucose-specific
        if patient.glucose > 140:
            recommendations.append("High glucose detected - dietary modifications recommended")
        elif patient.glucose < 70:
            recommendations.append("Low glucose detected - consult about hypoglycemia")
        
        # BMI-specific
        if patient.bmi > 30:
            recommendations.append("BMI indicates obesity - weight management recommended")
            recommendations.append("Consult nutritionist for diet plan")
        elif patient.bmi > 25:
            recommendations.append("BMI indicates overweight - consider lifestyle modifications")
        
        # Age-specific
        if patient.age > 45:
            recommendations.append("Regular health screenings recommended for your age group")
        
        # Blood pressure
        if patient.blood_pressure > 90:
            recommendations.append("Elevated blood pressure - monitor regularly")
        
        # General healthy lifestyle
        if prediction == 0 and probability < 0.3:
            recommendations.append("Maintain healthy diet and regular exercise")
            recommendations.append("Continue current healthy lifestyle habits")
        
        # If no specific recommendations
        if not recommendations:
            recommendations.append("Maintain regular health checkups")
            recommendations.append("Follow a balanced diet and exercise routine")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        try:
            # Load metadata if exists
            metadata_path = settings.model.model_path.with_suffix('.json')
            
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Add current info
            info = {
                'model_version': settings.model.version,
                'model_architecture': settings.model.architecture,
                'model_path': str(settings.model.model_path),
                'scaler_path': str(settings.model.scaler_path),
                'input_features': [
                    'Pregnancies', 'Glucose', 'BloodPressure',
                    'SkinThickness', 'Insulin', 'BMI',
                    'DiabetesPedigreeFunction', 'Age'
                ],
                'training_metrics': metadata.get('training_metrics', {}),
                'loaded_at': datetime.utcnow().isoformat()
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}", exc_info=True)
            return {
                'model_version': settings.model.version,
                'error': 'Failed to load metadata'
            }


# Create singleton instance (loaded once when app starts)
prediction_service = PredictionService()