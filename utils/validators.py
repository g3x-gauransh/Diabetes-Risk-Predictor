"""
Input validation utilities using Pydantic for type safety.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator, ValidationError
import numpy as np


class PatientData(BaseModel):
    """
    Patient data model with validation rules.
    All features must be within medically valid ranges.
    """
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=300, description="Plasma glucose concentration")
    blood_pressure: float = Field(..., ge=0, le=200, description="Diastolic blood pressure (mm Hg)")
    skin_thickness: float = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: float = Field(..., ge=0, le=900, description="2-Hour serum insulin (mu U/ml)")
    bmi: float = Field(..., ge=0, le=70, description="Body mass index (weight in kg/(height in m)^2)")
    diabetes_pedigree_function: float = Field(
        ..., 
        ge=0, 
        le=2.5, 
        description="Diabetes pedigree function"
    )
    age: int = Field(..., ge=0, le=120, description="Age in years")
    
    class Config:
        schema_extra = {
            "example": {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 100,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
        }
    
    @validator('glucose')
    def validate_glucose(cls, v):
        """Validate glucose levels are realistic"""
        if v > 0 and v < 40:
            raise ValueError('Glucose level too low for survival')
        if v > 250:
            import warnings
            warnings.warn('Glucose level critically high')
        return v
    
    @validator('blood_pressure')
    def validate_blood_pressure(cls, v):
        """Validate blood pressure is realistic"""
        if v > 0 and v < 40:
            raise ValueError('Blood pressure too low for survival')
        if v > 180:
            import warnings
            warnings.warn('Blood pressure critically high')
        return v
    
    @validator('bmi')
    def validate_bmi(cls, v):
        """Validate BMI is realistic"""
        if v > 0 and v < 10:
            raise ValueError('BMI too low for survival')
        if v > 60:
            import warnings
            warnings.warn('BMI critically high')
        return v
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([[
            self.pregnancies,
            self.glucose,
            self.blood_pressure,
            self.skin_thickness,
            self.insulin,
            self.bmi,
            self.diabetes_pedigree_function,
            self.age
        ]])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.dict()


class BatchPredictionRequest(BaseModel):
    """Model for batch prediction requests"""
    patients: List[PatientData] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        schema_extra = {
            "example": {
                "patients": [
                    {
                        "pregnancies": 6,
                        "glucose": 148,
                        "blood_pressure": 72,
                        "skin_thickness": 35,
                        "insulin": 100,
                        "bmi": 33.6,
                        "diabetes_pedigree_function": 0.627,
                        "age": 50
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Model for prediction response"""
    prediction: int = Field(..., ge=0, le=1, description="0 = No Diabetes, 1 = Diabetes")
    probability: float = Field(..., ge=0, le=1, description="Probability of diabetes")
    risk_level: str = Field(..., description="Low, Medium, or High")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    recommendations: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "risk_level": "High",
                "confidence": 0.85,
                "recommendations": [
                    "Consult with healthcare provider",
                    "Monitor blood glucose levels",
                    "Maintain healthy diet and exercise"
                ]
            }
        }


def validate_patient_data(data: Dict[str, Any]) -> PatientData:
    """
    Validate patient data and return PatientData instance.
    
    Args:
        data: Dictionary containing patient data
        
    Returns:
        Validated PatientData instance
        
    Raises:
        DataValidationError: If validation fails
    """
    from utils.exceptions import DataValidationError
    
    try:
        return PatientData(**data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = error['loc'][0]
            message = error['msg']
            errors.append(f"{field}: {message}")
        
        raise DataValidationError(
            message=f"Invalid patient data: {'; '.join(errors)}",
            field=None
        )