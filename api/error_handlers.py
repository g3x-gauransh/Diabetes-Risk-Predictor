"""
Centralized error handling for Flask application.

Similar to Spring's @ControllerAdvice.
"""

from flask import Flask, jsonify

from config.logging_config import logger
from utils.exceptions import DiabetesPredictorException


def register_error_handlers(app: Flask) -> None:
    """
    Register error handlers for the application.

    Args:
        app: Flask application instance
    """

    @app.errorhandler(DiabetesPredictorException)
    def handle_app_exception(error: DiabetesPredictorException):
        """Handle custom application exceptions."""
        logger.error(
            f"Application error: {error.message}",
            extra={"error_code": error.error_code},
        )

        return jsonify({"error": error.message, "error_code": error.error_code}), 500

    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        return (
            jsonify(
                {
                    "error": "Endpoint not found",
                    "error_code": "NOT_FOUND",
                    "path": (
                        error.description
                        if hasattr(error, "description")
                        else "unknown"
                    ),
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Handle 405 errors (wrong HTTP method)."""
        return (
            jsonify(
                {"error": "Method not allowed", "error_code": "METHOD_NOT_ALLOWED"}
            ),
            405,
        )

    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Handle rate limit errors."""
        return (
            jsonify(
                {
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": "Too many requests. Please try again later.",
                }
            ),
            429,
        )

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}", exc_info=True)
        return (
            jsonify({"error": "Internal server error", "error_code": "INTERNAL_ERROR"}),
            500,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Handle any unexpected errors."""
        logger.error(f"Unexpected error: {error}", exc_info=True)
        return (
            jsonify(
                {
                    "error": "An unexpected error occurred",
                    "error_code": "UNEXPECTED_ERROR",
                }
            ),
            500,
        )

    logger.info("✓ Error handlers registered")
