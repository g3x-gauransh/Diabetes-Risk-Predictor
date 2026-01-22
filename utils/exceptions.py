"""
Custom exception classes for the application.
"""


class DiabetesPredictorException(Exception):
    """Base exception for all application errors"""

    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        super().__init__(self.message)


class DataValidationError(DiabetesPredictorException):
    """Raised when data validation fails"""

    def __init__(self, message: str, field: str = None):
        self.field = field
        super().__init__(message, error_code="DATA_VALIDATION_ERROR")


class ModelNotFoundError(DiabetesPredictorException):
    """Raised when model file is not found"""

    def __init__(self, model_path: str):
        super().__init__(
            f"Model not found at path: {model_path}", error_code="MODEL_NOT_FOUND"
        )


class ModelLoadError(DiabetesPredictorException):
    """Raised when model fails to load"""

    def __init__(self, message: str):
        super().__init__(message, error_code="MODEL_LOAD_ERROR")


class PredictionError(DiabetesPredictorException):
    """Raised when prediction fails"""

    def __init__(self, message: str):
        super().__init__(message, error_code="PREDICTION_ERROR")


class TrainingError(DiabetesPredictorException):
    """Raised when model training fails"""

    def __init__(self, message: str):
        super().__init__(message, error_code="TRAINING_ERROR")


class ConfigurationError(DiabetesPredictorException):
    """Raised when configuration is invalid"""

    def __init__(self, message: str):
        super().__init__(message, error_code="CONFIGURATION_ERROR")


class AuthenticationError(DiabetesPredictorException):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTHENTICATION_ERROR")


class RateLimitExceeded(DiabetesPredictorException):
    """Raised when rate limit is exceeded"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED")
