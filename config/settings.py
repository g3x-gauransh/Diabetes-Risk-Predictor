"""
Configuration management using environment variables and config files.
Supports multiple environments (dev, staging, production).
"""
import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "diabetes_risk_predictor"
    version: str = os.getenv('MODEL_VERSION', '1.0.0')
    architecture: str = "neural_network"
    
    # Model hyperparameters
    input_dim: int = 8
    hidden_layers: list = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 200
    early_stopping_patience: int = 20
    
    # Model paths
    model_dir: Path = PROJECT_ROOT / 'artifacts' / 'models'
    scaler_dir: Path = PROJECT_ROOT / 'artifacts' / 'scalers'
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [64, 32, 16]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_path(self) -> Path:
        """Get current model path"""
        return self.model_dir / f"{self.model_name}_v{self.version}.h5"
    
    @property
    def scaler_path(self) -> Path:
        """Get current scaler path"""
        return self.scaler_dir / f"scaler_v{self.version}.pkl"


@dataclass
class APIConfig:
    """API configuration"""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', 5000))
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    workers: int = int(os.getenv('WORKERS', 4))
    
    # Security
    api_key_enabled: bool = os.getenv('API_KEY_ENABLED', 'False').lower() == 'true'
    api_key: str = os.getenv('API_KEY', '')
    
    # Rate limiting
    rate_limit_enabled: bool = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    rate_limit_per_minute: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', 60))
    
    # CORS
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_dir: Path = PROJECT_ROOT / 'logs'
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: Path = PROJECT_ROOT / 'data'
    raw_data_file: str = 'diabetes.csv'
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / self.raw_data_file


class Settings:
    """Main settings class combining all configurations"""
    
    def __init__(self, environment: str = None):
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        
        self.model = ModelConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.data = DataConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary project directories"""
        directories = [
            self.data.data_dir,
            self.model.model_dir,
            self.model.scaler_dir,
            self.logging.log_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'environment': self.environment,
            'model': {k: str(v) if isinstance(v, Path) else v 
                     for k, v in self.model.__dict__.items()},
            'api': self.api.__dict__,
            'logging': {k: str(v) if isinstance(v, Path) else v 
                       for k, v in self.logging.__dict__.items()},
            'data': {k: str(v) if isinstance(v, Path) else v 
                    for k, v in self.data.__dict__.items()},
        }
    
    @property
    def is_production(self) -> bool:
        return self.environment == 'production'
    
    @property
    def is_development(self) -> bool:
        return self.environment == 'development'


# IMPORTANT: Global settings instance - THIS MUST BE AT THE END
settings = Settings()