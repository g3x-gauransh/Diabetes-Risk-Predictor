"""
API routes and endpoints.

Defines all REST API endpoints for the application.
Similar to Spring's @RestController.
"""

from flask import Flask, request, jsonify
from typing import Dict, Any
import time

from config.logging_config import logger
from services.prediction_service import prediction_service
from utils.validators import validate_patient_data, BatchPredictionRequest
from utils.exceptions import DataValidationError, PredictionError


def register_routes(app: Flask) -> None:
    """
    Register all API routes.

    Args:
        app: Flask application instance
    """

    @app.route("/", methods=["GET"])
    def index() -> Dict[str, str]:
        """
        Root endpoint - API information.
        ---
        tags:
          - Health
        responses:
          200:
            description: API information
            schema:
              type: object
              properties:
                message:
                  type: string
                version:
                  type: string
                docs:
                  type: string
        """
        return jsonify(
            {
                "message": "Diabetes Risk Prediction API",
                "version": "1.0.0",
                "docs": "/docs",
                "endpoints": {
                    "predict_single": "/api/v1/predict",
                    "predict_batch": "/api/v1/predict/batch",
                    "model_info": "/api/v1/model/info",
                    "health_check": "/api/v1/health",
                },
            }
        )

    @app.route("/api/v1/health", methods=["GET"])
    def health_check() -> tuple[Dict[str, Any], int]:
        """
        Health check endpoint.

        Used by load balancers and monitoring systems.
        ---
        tags:
          - Health
        responses:
          200:
            description: Service is healthy
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: "healthy"
                timestamp:
                  type: string
                model_loaded:
                  type: boolean
        """
        from datetime import datetime

        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model_loaded": prediction_service.model is not None,
            "scaler_loaded": prediction_service.preprocessor is not None,
        }

        return jsonify(health_status), 200

    @app.route("/api/v1/model/info", methods=["GET"])
    def model_info() -> tuple[Dict[str, Any], int]:
        """
        Get model information and metadata.
        ---
        tags:
          - Model
        responses:
          200:
            description: Model information
            schema:
              type: object
              properties:
                model_version:
                  type: string
                model_architecture:
                  type: string
                training_metrics:
                  type: object
        """
        try:
            info = prediction_service.get_model_info()
            return jsonify(info), 200

        except Exception as e:
            logger.error(f"Failed to get model info: {e}", exc_info=True)
            return jsonify({"error": "Failed to retrieve model information"}), 500

    @app.route("/api/v1/predict", methods=["POST"])
    def predict_single() -> tuple[Dict[str, Any], int]:
        """
        Predict diabetes risk for a single patient.
        ---
        tags:
          - Prediction
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              required:
                - pregnancies
                - glucose
                - blood_pressure
                - skin_thickness
                - insulin
                - bmi
                - diabetes_pedigree_function
                - age
              properties:
                pregnancies:
                  type: integer
                  minimum: 0
                  maximum: 20
                  example: 6
                glucose:
                  type: number
                  minimum: 0
                  maximum: 300
                  example: 148
                blood_pressure:
                  type: number
                  minimum: 0
                  maximum: 200
                  example: 72
                skin_thickness:
                  type: number
                  minimum: 0
                  maximum: 100
                  example: 35
                insulin:
                  type: number
                  minimum: 0
                  maximum: 900
                  example: 100
                bmi:
                  type: number
                  minimum: 0
                  maximum: 70
                  example: 33.6
                diabetes_pedigree_function:
                  type: number
                  minimum: 0
                  maximum: 2.5
                  example: 0.627
                age:
                  type: integer
                  minimum: 0
                  maximum: 120
                  example: 50
        responses:
          200:
            description: Successful prediction
            schema:
              type: object
              properties:
                prediction:
                  type: integer
                  description: "0 = No Diabetes, 1 = Diabetes"
                probability:
                  type: number
                  description: "Probability of diabetes (0-1)"
                risk_level:
                  type: string
                  enum: ["Low", "Medium", "High"]
                confidence:
                  type: number
                recommendations:
                  type: array
                  items:
                    type: string
          400:
            description: Invalid input data
          500:
            description: Internal server error
        """
        start_time = time.time()

        try:
            # Get JSON data from request
            data = request.get_json()

            if not data:
                return (
                    jsonify({"error": "No data provided", "error_code": "NO_DATA"}),
                    400,
                )

            # Validate input using Pydantic
            patient_data = validate_patient_data(data)

            # Make prediction
            response = prediction_service.predict_single(patient_data)

            # Calculate request duration
            duration_ms = (time.time() - start_time) * 1000

            logger.info(f"Prediction request completed in {duration_ms:.2f}ms")

            # Return response
            result = response.dict()
            result["request_duration_ms"] = round(duration_ms, 2)

            return jsonify(result), 200

        except DataValidationError as e:
            logger.warning(f"Validation error: {e.message}", extra={"field": e.field})
            return (
                jsonify(
                    {"error": e.message, "error_code": e.error_code, "field": e.field}
                ),
                400,
            )

        except PredictionError as e:
            logger.error(f"Prediction error: {e.message}")
            return (
                jsonify(
                    {"error": "Failed to make prediction", "error_code": e.error_code}
                ),
                500,
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return (
                jsonify(
                    {"error": "Internal server error", "error_code": "INTERNAL_ERROR"}
                ),
                500,
            )

    @app.route("/api/v1/predict/batch", methods=["POST"])
    def predict_batch() -> tuple[Dict[str, Any], int]:
        """
        Predict diabetes risk for multiple patients.
        ---
        tags:
          - Prediction
        parameters:
          - in: body
            name: body
            required: true
            schema:
              type: object
              properties:
                patients:
                  type: array
                  items:
                    type: object
                  minItems: 1
                  maxItems: 1000
        responses:
          200:
            description: Successful batch prediction
            schema:
              type: object
              properties:
                predictions:
                  type: array
                  items:
                    type: object
                count:
                  type: integer
          400:
            description: Invalid input
          500:
            description: Internal server error
        """
        start_time = time.time()

        try:
            data = request.get_json()

            if not data or "patients" not in data:
                return (
                    jsonify(
                        {"error": "No patient data provided", "error_code": "NO_DATA"}
                    ),
                    400,
                )

            # Validate batch request
            batch_request = BatchPredictionRequest(**data)

            # Make predictions
            responses = prediction_service.predict_batch(batch_request.patients)

            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Batch prediction completed in {duration_ms:.2f}ms for {len(responses)} patients"
            )

            return (
                jsonify(
                    {
                        "predictions": [r.dict() for r in responses],
                        "count": len(responses),
                        "request_duration_ms": round(duration_ms, 2),
                    }
                ),
                200,
            )

        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            return (
                jsonify(
                    {
                        "error": "Failed to process batch prediction",
                        "error_code": "BATCH_PREDICTION_ERROR",
                    }
                ),
                500,
            )
