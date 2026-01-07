"""
Centralized logging configuration with structured logging support.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        return json.dumps(log_data)


def setup_logging(
    log_level: str = 'INFO',
    log_dir: Optional[Path] = None,
    log_file: str = 'app.log',
    structured: bool = False
) -> logging.Logger:
    """
    Setup application logging with file and console handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_file: Name of log file
        structured: Use structured JSON logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('diabetes_predictor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler (rotating)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
    
    # Set formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_dir:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# IMPORTANT: Create global logger instance - THIS MUST BE AT THE END
# Import settings here to avoid circular import
try:
    from config.settings import settings
    logger = setup_logging(
        log_level=settings.logging.level,      # ✓ Correct path
        log_dir=settings.logging.log_dir,      # ✓ Correct path
        structured=settings.is_production
    )
except Exception as e:
    # Fallback if settings import fails
    logger = setup_logging(
        log_level='INFO',
        log_dir=Path(__file__).parent.parent / 'logs',
        structured=False
    )
    logger.warning(f"Failed to load settings for logging: {e}")