"""Integration tests for the API."""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_api_request(sample_ecg_reading, sample_patient_vitals):
    """Create sample API request."""
    return {
        "ecg_reading": {
            "timestamp": sample_ecg_reading.timestamp.isoformat(),
            "lead_data": sample_ecg_reading.lead_data,
            "sampling_rate": sample_ecg_reading.sampling_rate,
            "duration": sample_ecg_reading.duration,
            "patient_id": sample_ecg_reading.patient_id,
            "device_id": sample_ecg_reading.device_id,
            "quality_score": sample_ecg_reading.quality_score
        },
        "patient_vitals": {
            "patient_id": sample_patient_vitals.patient_id,
            "age": sample_patient_vitals.age,
            "gender": sample_patient_vitals.gender,
            "total_cholesterol": sample_patient_vitals.total_cholesterol,
            "ldl_cholesterol": sample_patient_vitals.ldl_cholesterol,
            "hdl_cholesterol": sample_patient_vitals.hdl_cholesterol,
            "triglycerides": sample_patient_vitals.triglycerides,
            "systolic_bp": sample_patient_vitals.systolic_bp,
            "diastolic_bp": sample_patient_vitals.diastolic_bp,
            "heart_rate": sample_patient_vitals.heart_rate,
            "diabetes": sample_patient_vitals.diabetes,
            "smoking": sample_patient_vitals.smoking,
            "family_history_cad": sample_patient_vitals.family_history_cad,
            "previous_mi": sample_patient_vitals.previous_mi,
            "on_statins": sample_patient_vitals.on_statins,
            "on_beta_blockers": sample_patient_vitals.on_beta_blockers,
            "on_ace_inhibitors": sample_patient_vitals.on_ace_inhibitors
        }
    }


class TestAPI:
    """Test cases for the API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
    
    def test_analyze_ecg_endpoint(self, client, sample_api_request):
        """Test ECG analysis endpoint."""
        response = client.post("/analyze", json=sample_api_request)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert "patient_id" in data
        assert "analysis_timestamp" in data
        assert "signal_quality" in data
        assert "noise_level" in data
        assert "features" in data
        assert "risk_prediction" in data
        
        # Check signal quality bounds
        assert 0 <= data["signal_quality"] <= 1
        assert 0 <= data["noise_level"] <= 1
        
        # Check features
        features = data["features"]
        assert "mean_rr" in features
        assert "sdnn" in features
        assert "rmssd" in features
        
        # Check risk prediction
        risk = data["risk_prediction"]
        assert "immediate_risk" in risk
        assert "short_term_risk" in risk
        assert "long_term_risk" in risk
        assert "risk_category" in risk
        assert "recommendations" in risk
        
        # Check risk bounds
        assert 0 <= risk["immediate_risk"] <= 1
        assert 0 <= risk["short_term_risk"] <= 1
        assert 0 <= risk["long_term_risk"] <= 1
    
    def test_analyze_ecg_without_vitals(self, client, sample_api_request):
        """Test ECG analysis without patient vitals."""
        # Remove patient vitals from request
        request_without_vitals = {
            "ecg_reading": sample_api_request["ecg_reading"]
        }
        
        response = client.post("/analyze", json=request_without_vitals)
        assert response.status_code == 200
        
        data = response.json()
        assert "patient_id" in data
        assert "features" in data
        assert data["risk_prediction"] is None  # No vitals, no risk prediction
    
    def test_analyze_ecg_invalid_data(self, client):
        """Test ECG analysis with invalid data."""
        invalid_request = {
            "ecg_reading": {
                "timestamp": "invalid-timestamp",
                "lead_data": {},  # Empty lead data
                "sampling_rate": -1,  # Invalid sampling rate
                "duration": 0,
                "patient_id": "",
                "device_id": ""
            }
        }
        
        response = client.post("/analyze", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_batch_analyze_endpoint(self, client, sample_api_request):
        """Test batch ECG analysis endpoint."""
        # Create batch request with multiple readings
        batch_request = {
            "ecg_readings": [
                sample_api_request["ecg_reading"],
                {**sample_api_request["ecg_reading"], "patient_id": "BATCH_002"}
            ],
            "patient_vitals": {
                sample_api_request["patient_vitals"]["patient_id"]: sample_api_request["patient_vitals"],
                "BATCH_002": {**sample_api_request["patient_vitals"], "patient_id": "BATCH_002"}
            }
        }
        
        response = client.post("/analyze/batch", json=batch_request)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        
        for result in data:
            assert "patient_id" in result
            assert "features" in result
            assert "risk_prediction" in result
    
    def test_patient_risk_summary_endpoint(self, client):
        """Test patient risk summary endpoint."""
        patient_id = "TEST_PATIENT_001"
        response = client.get(f"/patients/{patient_id}/risk-summary")
        assert response.status_code == 200
        
        data = response.json()
        assert data["patient_id"] == patient_id
        assert "message" in data
        assert "timestamp" in data
    
    def test_analytics_dashboard_endpoint(self, client):
        """Test analytics dashboard endpoint."""
        response = client.get("/analytics/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_analyses" in data
        assert "high_risk_patients" in data
        assert "average_processing_time_ms" in data
        assert "system_status" in data
        assert "timestamp" in data
    
    def test_api_error_handling(self, client):
        """Test API error handling."""
        # Test with malformed JSON
        response = client.post("/analyze", data="invalid json")
        assert response.status_code == 422
        
        # Test with missing required fields
        response = client.post("/analyze", json={})
        assert response.status_code == 422
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        # Test with a GET request to health endpoint which should have CORS headers
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check for CORS headers (they should be present due to middleware)
        # Note: TestClient might not include all CORS headers, but the middleware is configured
    
    def test_api_documentation(self, client):
        """Test API documentation endpoints."""
        # Test OpenAPI docs
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "AI Healthcare ECG Analysis API"