"""FastAPI application for ECG analysis service."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import logging
import uvicorn

from ..models.ecg_data import ECGReading, PatientVitals, ECGAnalysisResult
from ..models.ecg_analyzer import ECGAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Healthcare ECG Analysis API",
    description="AI-powered ECG analysis for heart attack prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ECG analyzer
ecg_analyzer = ECGAnalyzer()

# Request/Response models
class ECGAnalysisRequest(BaseModel):
    """Request model for ECG analysis."""
    ecg_reading: ECGReading
    patient_vitals: Optional[PatientVitals] = None

class BatchECGAnalysisRequest(BaseModel):
    """Request model for batch ECG analysis."""
    ecg_readings: List[ECGReading]
    patient_vitals: Optional[Dict[str, PatientVitals]] = None

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str

@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with basic service information."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/analyze", response_model=ECGAnalysisResult)
async def analyze_ecg(request: ECGAnalysisRequest):
    """Analyze a single ECG reading.
    
    Args:
        request: ECG analysis request containing ECG data and optional patient vitals
        
    Returns:
        ECG analysis result with risk prediction
    """
    try:
        logger.info(f"Analyzing ECG for patient {request.ecg_reading.patient_id}")
        
        result = ecg_analyzer.analyze_ecg(
            ecg_reading=request.ecg_reading,
            patient_vitals=request.patient_vitals
        )
        
        logger.info(f"Analysis completed for patient {request.ecg_reading.patient_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing ECG: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch", response_model=List[ECGAnalysisResult])
async def analyze_ecg_batch(request: BatchECGAnalysisRequest):
    """Analyze multiple ECG readings in batch.
    
    Args:
        request: Batch ECG analysis request
        
    Returns:
        List of ECG analysis results
    """
    try:
        logger.info(f"Batch analyzing {len(request.ecg_readings)} ECG readings")
        
        results = ecg_analyzer.batch_analyze(
            ecg_readings=request.ecg_readings,
            patient_vitals=request.patient_vitals
        )
        
        logger.info(f"Batch analysis completed for {len(results)} readings")
        return results
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/patients/{patient_id}/risk-summary")
async def get_patient_risk_summary(patient_id: str):
    """Get risk summary for a specific patient.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Patient risk summary (placeholder for now)
    """
    # This would typically query a database for historical analysis results
    return {
        "patient_id": patient_id,
        "message": "Risk summary endpoint - would return historical analysis data",
        "timestamp": datetime.now()
    }

@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get analytics dashboard data.
    
    Returns:
        Dashboard analytics data (placeholder for now)
    """
    return {
        "total_analyses": 0,
        "high_risk_patients": 0,
        "average_processing_time_ms": 0,
        "system_status": "operational",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=12000,
        reload=True,
        log_level="info"
    )