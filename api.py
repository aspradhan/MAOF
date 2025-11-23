"""
Multi-Agent Orchestration Framework (MAOF)
FastAPI Application
Version: 3.0
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MAOF-API')

# Create FastAPI app
app = FastAPI(
    title="MAOF - Multi-Agent Orchestration Framework",
    description="Best practices framework for OrchaMesh and multi-agent systems",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for container orchestration.
    Returns service status and basic information.
    """
    return {
        "status": "healthy",
        "service": "MAOF",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "framework": "best-practices"
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint with framework information.
    """
    return {
        "name": "MAOF - Multi-Agent Orchestration Framework",
        "version": "3.0.0",
        "description": "Best practices framework for OrchaMesh and multi-agent systems",
        "documentation": {
            "openapi": "/docs",
            "redoc": "/redoc"
        },
        "health": "/health",
        "framework_type": "best-practices",
        "integration": "OrchaMesh"
    }


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Execute startup tasks."""
    logger.info("MAOF API starting up...")
    logger.info("Version: 3.0.0")
    logger.info("Framework type: Best Practices")


@app.on_event("shutdown")
async def shutdown_event():
    """Execute cleanup tasks."""
    logger.info("MAOF API shutting down...")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
