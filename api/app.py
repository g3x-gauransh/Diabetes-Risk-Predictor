"""
Flask application for diabetes risk prediction API.

This is the main entry point for the API server.
Similar to Spring Boot's main application class.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flasgger import Swagger, swag_from
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from config.logging_config import logger
from utils.exceptions import DiabetesPredictorException
from api.routes import register_routes
from api.error_handlers import register_error_handlers


def create_app() -> Flask:
    """
    Application factory pattern.
    
    Creates and configures Flask application.
    Similar to Spring Boot's application configuration.
    
    Returns:
        Configured Flask app
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Configure from settings
    app.config['DEBUG'] = settings.api.debug
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
    
    # CORS configuration (allow cross-origin requests)
    CORS(app, origins=settings.api.cors_origins)
    
    # Rate limiting (prevent abuse)
    if settings.api.rate_limit_enabled:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=[f"{settings.api.rate_limit_per_minute}/minute"],
            storage_uri="memory://"
        )
        logger.info(f"Rate limiting enabled: {settings.api.rate_limit_per_minute} requests/minute")
    
    # Swagger/OpenAPI documentation
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs"
    }
    
    swagger_template = {
        "info": {
            "title": "Diabetes Risk Prediction API",
            "description": "REST API for predicting diabetes risk using machine learning",
            "version": settings.model.version,
            "contact": {
                "name": "Gauransh Gupta",
                "email": "gauransh@northeastern.edu"
            }
        },
        "schemes": ["http", "https"],
        "tags": [
            {
                "name": "Prediction",
                "description": "Endpoints for making diabetes risk predictions"
            },
            {
                "name": "Health",
                "description": "API health and monitoring endpoints"
            },
            {
                "name": "Model",
                "description": "Model information and metadata"
            }
        ]
    }
    
    Swagger(app, config=swagger_config, template=swagger_template)
    
    # Register routes and error handlers
    register_routes(app)
    register_error_handlers(app)
    
    logger.info("="*60)
    logger.info("FLASK APPLICATION INITIALIZED")
    logger.info("="*60)
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.api.debug}")
    logger.info(f"API documentation: http://{settings.api.host}:{settings.api.port}/docs")
    logger.info("="*60)
    
    return app


# Create app instance
app = create_app()


if __name__ == '__main__':
    """
    Run Flask development server.
    
    For production, use: gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
    """
    logger.info(f"Starting Flask server on {settings.api.host}:{settings.api.port}")
    
    app.run(
        host=settings.api.host,
        port=settings.api.port,
        debug=settings.api.debug
    )