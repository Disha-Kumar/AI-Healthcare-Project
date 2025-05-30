"""Configuration settings for the ECG analysis system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 12000
    api_reload: bool = True
    api_log_level: str = "info"
    
    # ECG Processing Configuration
    default_sampling_rate: int = 500
    signal_quality_threshold: float = 0.6
    noise_level_threshold: float = 0.4
    
    # Risk Prediction Configuration
    risk_model_version: str = "1.0.0"
    confidence_threshold: float = 0.7
    urgent_risk_threshold: float = 0.7
    
    # Cholesterol Thresholds (mg/dL)
    high_total_cholesterol: float = 240.0
    high_ldl_cholesterol: float = 160.0
    low_hdl_cholesterol_male: float = 40.0
    low_hdl_cholesterol_female: float = 50.0
    high_triglycerides: float = 200.0
    
    # Blood Pressure Thresholds (mmHg)
    high_systolic_bp: int = 140
    high_diastolic_bp: int = 90
    very_high_systolic_bp: int = 180
    very_high_diastolic_bp: int = 110
    
    # ECG Analysis Thresholds
    bradycardia_threshold: int = 60  # bpm
    tachycardia_threshold: int = 100  # bpm
    qt_prolongation_threshold: float = 450.0  # ms
    qrs_widening_threshold: float = 120.0  # ms
    st_elevation_threshold: float = 0.1  # mV
    st_depression_threshold: float = 0.1  # mV
    
    # HRV Thresholds
    low_hrv_sdnn_threshold: float = 20.0  # ms
    high_hrv_sdnn_threshold: float = 100.0  # ms
    irregular_rhythm_threshold: float = 150.0  # ms
    
    # Database Configuration (for future use)
    database_url: Optional[str] = None
    redis_url: Optional[str] = None
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    api_key_header: str = "X-API-Key"
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]
    
    # Model Configuration
    model_cache_dir: str = "models/cache"
    model_update_interval: int = 3600  # seconds
    
    # Data Validation
    max_ecg_duration: float = 300.0  # seconds
    min_ecg_duration: float = 5.0  # seconds
    max_sampling_rate: int = 2000  # Hz
    min_sampling_rate: int = 100  # Hz
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


# Risk category mappings
RISK_CATEGORIES = {
    "Low": {"color": "green", "priority": 1},
    "Moderate": {"color": "yellow", "priority": 2},
    "High": {"color": "orange", "priority": 3},
    "Critical": {"color": "red", "priority": 4}
}

# ECG Lead configurations
ECG_LEADS = {
    "standard": ["I", "II", "III", "aVR", "aVL", "aVF"],
    "precordial": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "minimal": ["I", "II", "V1"]
}

# Medication categories for risk reduction
PROTECTIVE_MEDICATIONS = {
    "statins": {"risk_reduction": 0.1, "cholesterol_target": True},
    "beta_blockers": {"risk_reduction": 0.05, "heart_rate_control": True},
    "ace_inhibitors": {"risk_reduction": 0.05, "bp_control": True},
    "aspirin": {"risk_reduction": 0.03, "antiplatelet": True}
}

# Age-based risk multipliers
AGE_RISK_MULTIPLIERS = {
    (0, 30): 0.5,
    (30, 40): 0.7,
    (40, 50): 1.0,
    (50, 60): 1.3,
    (60, 70): 1.6,
    (70, 80): 2.0,
    (80, 100): 2.5
}

# Gender-based risk adjustments
GENDER_RISK_ADJUSTMENTS = {
    "M": 1.2,  # Males have higher baseline risk
    "F": 1.0,
    "Other": 1.1
}