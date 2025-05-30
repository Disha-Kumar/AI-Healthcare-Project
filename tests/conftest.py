"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict, List

from src.models.ecg_data import ECGReading, PatientVitals, ECGFeatures
from src.models.ecg_analyzer import ECGAnalyzer
from src.utils.ecg_processor import ECGProcessor


@pytest.fixture
def sample_ecg_data():
    """Generate sample ECG data for testing."""
    # Generate synthetic ECG signal (simplified)
    duration = 10  # seconds
    sampling_rate = 500  # Hz
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Simple synthetic ECG with QRS complexes
    heart_rate = 70  # bpm
    rr_interval = 60 / heart_rate  # seconds
    
    ecg_signal = []
    for i in range(len(t)):
        # Create periodic QRS-like complexes
        phase = (t[i] % rr_interval) / rr_interval * 2 * np.pi
        
        # QRS complex (simplified)
        if 0.4 * np.pi < phase < 0.6 * np.pi:
            amplitude = 1.0 * np.sin((phase - 0.4 * np.pi) * 5)
        else:
            amplitude = 0.1 * np.sin(phase) + 0.05 * np.random.normal()
        
        ecg_signal.append(amplitude)
    
    return {
        'I': ecg_signal,
        'II': ecg_signal,
        'V1': [x * 0.8 for x in ecg_signal]
    }


@pytest.fixture
def sample_ecg_reading(sample_ecg_data):
    """Create a sample ECG reading."""
    return ECGReading(
        timestamp=datetime.now(),
        lead_data=sample_ecg_data,
        sampling_rate=500,
        duration=10.0,
        patient_id="TEST_001",
        device_id="ECG_DEVICE_001",
        quality_score=0.85
    )


@pytest.fixture
def sample_patient_vitals():
    """Create sample patient vitals."""
    return PatientVitals(
        patient_id="TEST_001",
        age=55,
        gender="M",
        total_cholesterol=220.0,
        ldl_cholesterol=140.0,
        hdl_cholesterol=45.0,
        triglycerides=180.0,
        systolic_bp=140,
        diastolic_bp=90,
        heart_rate=75,
        diabetes=False,
        smoking=True,
        family_history_cad=True,
        previous_mi=False,
        on_statins=False,
        on_beta_blockers=False,
        on_ace_inhibitors=False
    )


@pytest.fixture
def high_risk_patient_vitals():
    """Create high-risk patient vitals."""
    return PatientVitals(
        patient_id="HIGH_RISK_001",
        age=70,
        gender="M",
        total_cholesterol=300.0,
        ldl_cholesterol=200.0,
        hdl_cholesterol=35.0,
        triglycerides=250.0,
        systolic_bp=180,
        diastolic_bp=100,
        heart_rate=85,
        diabetes=True,
        smoking=True,
        family_history_cad=True,
        previous_mi=True,
        on_statins=False,
        on_beta_blockers=False,
        on_ace_inhibitors=False
    )


@pytest.fixture
def low_risk_patient_vitals():
    """Create low-risk patient vitals."""
    return PatientVitals(
        patient_id="LOW_RISK_001",
        age=35,
        gender="F",
        total_cholesterol=180.0,
        ldl_cholesterol=100.0,
        hdl_cholesterol=60.0,
        triglycerides=120.0,
        systolic_bp=110,
        diastolic_bp=70,
        heart_rate=65,
        diabetes=False,
        smoking=False,
        family_history_cad=False,
        previous_mi=False,
        on_statins=True,
        on_beta_blockers=False,
        on_ace_inhibitors=False
    )


@pytest.fixture
def ecg_processor():
    """Create ECG processor instance."""
    return ECGProcessor(sampling_rate=500)


@pytest.fixture
def ecg_analyzer():
    """Create ECG analyzer instance."""
    return ECGAnalyzer(sampling_rate=500)


@pytest.fixture
def sample_ecg_features():
    """Create sample ECG features."""
    return ECGFeatures(
        mean_rr=857.0,  # ~70 bpm
        sdnn=45.0,
        rmssd=35.0,
        pnn50=15.0,
        lf_power=500.0,
        hf_power=300.0,
        lf_hf_ratio=1.67,
        p_wave_duration=100.0,
        pr_interval=160.0,
        qrs_duration=100.0,
        qt_interval=400.0,
        qtc_interval=420.0,
        st_elevation={'II': 0.0},
        st_depression={'II': 0.0},
        t_wave_amplitude={'II': 0.3},
        t_wave_inversion={'II': False}
    )


@pytest.fixture
def abnormal_ecg_features():
    """Create abnormal ECG features for testing."""
    return ECGFeatures(
        mean_rr=600.0,  # ~100 bpm (tachycardia)
        sdnn=15.0,      # Low HRV
        rmssd=20.0,
        pnn50=5.0,
        lf_power=200.0,
        hf_power=100.0,
        lf_hf_ratio=2.0,
        p_wave_duration=120.0,
        pr_interval=200.0,
        qrs_duration=130.0,  # Wide QRS
        qt_interval=480.0,
        qtc_interval=520.0,  # QT prolongation
        st_elevation={'II': 0.25},  # ST elevation
        st_depression={'II': 0.0},
        t_wave_amplitude={'II': 0.1},
        t_wave_inversion={'II': True}  # T wave inversion
    )