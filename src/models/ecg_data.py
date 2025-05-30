"""ECG data models and structures."""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
from pydantic import BaseModel, Field, field_validator


@dataclass
class ECGReading:
    """Raw ECG reading data structure."""
    
    timestamp: datetime
    lead_data: Dict[str, List[float]]  # e.g., {'I': [...], 'II': [...], 'V1': [...]}
    sampling_rate: int  # Hz
    duration: float  # seconds
    patient_id: str
    device_id: str
    quality_score: Optional[float] = None


class PatientVitals(BaseModel):
    """Patient vital signs and risk factors."""
    
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150, description="Patient age in years")
    gender: str = Field(..., pattern="^(M|F|Other)$", description="Patient gender")
    
    # Cholesterol levels (mg/dL)
    total_cholesterol: Optional[float] = Field(None, ge=0, le=1000)
    ldl_cholesterol: Optional[float] = Field(None, ge=0, le=500)
    hdl_cholesterol: Optional[float] = Field(None, ge=0, le=200)
    triglycerides: Optional[float] = Field(None, ge=0, le=2000)
    
    # Other cardiovascular risk factors
    systolic_bp: Optional[int] = Field(None, ge=50, le=300, description="Systolic blood pressure")
    diastolic_bp: Optional[int] = Field(None, ge=30, le=200, description="Diastolic blood pressure")
    heart_rate: Optional[int] = Field(None, ge=30, le=250, description="Resting heart rate")
    
    # Medical history
    diabetes: bool = Field(False, description="History of diabetes")
    smoking: bool = Field(False, description="Current or former smoker")
    family_history_cad: bool = Field(False, description="Family history of coronary artery disease")
    previous_mi: bool = Field(False, description="Previous myocardial infarction")
    
    # Medications
    on_statins: bool = Field(False, description="Currently on statin therapy")
    on_beta_blockers: bool = Field(False, description="Currently on beta blockers")
    on_ace_inhibitors: bool = Field(False, description="Currently on ACE inhibitors")
    
    @field_validator('ldl_cholesterol')
    @classmethod
    def validate_ldl_cholesterol(cls, v):
        """Validate LDL cholesterol levels."""
        if v is not None and v > 190:
            # High LDL cholesterol warning
            pass
        return v
    
    @field_validator('hdl_cholesterol')
    @classmethod
    def validate_hdl_cholesterol(cls, v, info):
        """Validate HDL cholesterol levels."""
        if v is not None:
            gender = info.data.get('gender')
            if gender == 'M' and v < 40:
                # Low HDL for men
                pass
            elif gender == 'F' and v < 50:
                # Low HDL for women
                pass
        return v


class ECGFeatures(BaseModel):
    """Extracted ECG features for analysis."""
    
    # Heart rate variability features
    mean_rr: float = Field(..., description="Mean RR interval (ms)")
    sdnn: float = Field(..., description="Standard deviation of NN intervals")
    rmssd: float = Field(..., description="Root mean square of successive differences")
    pnn50: float = Field(..., description="Percentage of NN intervals > 50ms different")
    
    # Frequency domain features
    lf_power: float = Field(..., description="Low frequency power (0.04-0.15 Hz)")
    hf_power: float = Field(..., description="High frequency power (0.15-0.4 Hz)")
    lf_hf_ratio: float = Field(..., description="LF/HF ratio")
    
    # Morphological features
    p_wave_duration: Optional[float] = Field(None, description="P wave duration (ms)")
    pr_interval: Optional[float] = Field(None, description="PR interval (ms)")
    qrs_duration: Optional[float] = Field(None, description="QRS duration (ms)")
    qt_interval: Optional[float] = Field(None, description="QT interval (ms)")
    qtc_interval: Optional[float] = Field(None, description="QT corrected interval (ms)")
    
    # ST segment analysis
    st_elevation: Dict[str, float] = Field(default_factory=dict, description="ST elevation by lead")
    st_depression: Dict[str, float] = Field(default_factory=dict, description="ST depression by lead")
    
    # T wave features
    t_wave_amplitude: Dict[str, float] = Field(default_factory=dict, description="T wave amplitude by lead")
    t_wave_inversion: Dict[str, bool] = Field(default_factory=dict, description="T wave inversion by lead")


class RiskPrediction(BaseModel):
    """Heart attack risk prediction results."""
    
    patient_id: str
    prediction_timestamp: datetime
    
    # Risk scores (0-1 probability)
    immediate_risk: float = Field(..., ge=0, le=1, description="Risk within 24 hours")
    short_term_risk: float = Field(..., ge=0, le=1, description="Risk within 30 days")
    long_term_risk: float = Field(..., ge=0, le=1, description="Risk within 1 year")
    
    # Risk category
    risk_category: str = Field(..., pattern="^(Low|Moderate|High|Critical)$")
    
    # Contributing factors
    ecg_risk_score: float = Field(..., ge=0, le=1, description="ECG-based risk component")
    cholesterol_risk_score: float = Field(..., ge=0, le=1, description="Cholesterol-based risk component")
    clinical_risk_score: float = Field(..., ge=0, le=1, description="Clinical factors risk component")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    urgent_action_required: bool = Field(False)
    
    # Model metadata
    model_version: str = Field(..., description="Version of the prediction model used")
    confidence_score: float = Field(..., ge=0, le=1, description="Model confidence in prediction")


class ECGAnalysisResult(BaseModel):
    """Complete ECG analysis result."""
    
    patient_id: str
    analysis_timestamp: datetime
    ecg_reading_id: str
    
    # Quality assessment
    signal_quality: float = Field(..., ge=0, le=1, description="Overall signal quality score")
    noise_level: float = Field(..., ge=0, le=1, description="Noise level in signal")
    
    # Extracted features
    features: ECGFeatures
    
    # Abnormality detection
    rhythm_abnormalities: List[str] = Field(default_factory=list)
    morphology_abnormalities: List[str] = Field(default_factory=list)
    
    # Risk prediction
    risk_prediction: Optional[RiskPrediction] = None
    
    # Processing metadata
    processing_time_ms: float = Field(..., description="Time taken for analysis")
    algorithm_version: str = Field(..., description="Version of analysis algorithm")