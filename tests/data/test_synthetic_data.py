"""Test synthetic data generation for ECG analysis."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import random

from src.models.ecg_data import ECGReading, PatientVitals


class SyntheticDataGenerator:
    """Generate synthetic ECG and patient data for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_ecg_signal(self, duration: float = 10.0, sampling_rate: int = 500, 
                           heart_rate: float = 70.0, noise_level: float = 0.1) -> np.ndarray:
        """Generate synthetic ECG signal.
        
        Args:
            duration: Signal duration in seconds
            sampling_rate: Sampling rate in Hz
            heart_rate: Heart rate in BPM
            noise_level: Noise level (0-1)
            
        Returns:
            Synthetic ECG signal
        """
        t = np.linspace(0, duration, int(duration * sampling_rate))
        rr_interval = 60.0 / heart_rate  # seconds
        
        ecg_signal = np.zeros(len(t))
        
        # Generate QRS complexes
        for beat_time in np.arange(0, duration, rr_interval):
            beat_idx = int(beat_time * sampling_rate)
            
            if beat_idx < len(t):
                # P wave (small positive deflection)
                p_start = max(0, beat_idx - int(0.15 * sampling_rate))
                p_end = max(0, beat_idx - int(0.05 * sampling_rate))
                if p_end > p_start:
                    p_wave = 0.2 * np.exp(-((np.arange(p_end - p_start) - (p_end - p_start)/2) ** 2) / (0.02 * sampling_rate))
                    ecg_signal[p_start:p_end] += p_wave
                
                # QRS complex (main deflection)
                qrs_start = max(0, beat_idx - int(0.04 * sampling_rate))
                qrs_end = min(len(t), beat_idx + int(0.04 * sampling_rate))
                if qrs_end > qrs_start:
                    qrs_wave = np.exp(-((np.arange(qrs_end - qrs_start) - (qrs_end - qrs_start)/2) ** 2) / (0.01 * sampling_rate))
                    ecg_signal[qrs_start:qrs_end] += qrs_wave
                
                # T wave (positive deflection after QRS)
                t_start = min(len(t), beat_idx + int(0.1 * sampling_rate))
                t_end = min(len(t), beat_idx + int(0.3 * sampling_rate))
                if t_end > t_start:
                    t_wave = 0.3 * np.exp(-((np.arange(t_end - t_start) - (t_end - t_start)/2) ** 2) / (0.05 * sampling_rate))
                    ecg_signal[t_start:t_end] += t_wave
        
        # Add baseline noise
        noise = np.random.normal(0, noise_level, len(t))
        ecg_signal += noise
        
        return ecg_signal
    
    def generate_abnormal_ecg(self, duration: float = 10.0, sampling_rate: int = 500,
                             abnormality_type: str = "st_elevation") -> np.ndarray:
        """Generate ECG with specific abnormalities.
        
        Args:
            duration: Signal duration in seconds
            sampling_rate: Sampling rate in Hz
            abnormality_type: Type of abnormality to simulate
            
        Returns:
            Abnormal ECG signal
        """
        # Start with normal ECG
        if abnormality_type == "tachycardia":
            ecg_signal = self.generate_ecg_signal(duration, sampling_rate, heart_rate=120)
        elif abnormality_type == "bradycardia":
            ecg_signal = self.generate_ecg_signal(duration, sampling_rate, heart_rate=45)
        else:
            ecg_signal = self.generate_ecg_signal(duration, sampling_rate, heart_rate=70)
        
        # Add specific abnormalities
        if abnormality_type == "st_elevation":
            # Add ST elevation
            ecg_signal += 0.3  # Elevate entire signal
        elif abnormality_type == "st_depression":
            # Add ST depression
            ecg_signal -= 0.2  # Depress entire signal
        elif abnormality_type == "irregular_rhythm":
            # Add irregular rhythm by varying RR intervals
            t = np.linspace(0, duration, int(duration * sampling_rate))
            irregular_component = 0.1 * np.sin(2 * np.pi * 0.1 * t) * np.random.normal(1, 0.3, len(t))
            ecg_signal += irregular_component
        elif abnormality_type == "high_noise":
            # Add high noise level
            noise = np.random.normal(0, 0.5, len(ecg_signal))
            ecg_signal += noise
        
        return ecg_signal
    
    def generate_ecg_reading(self, patient_id: str = None, abnormality: str = None) -> ECGReading:
        """Generate complete ECG reading.
        
        Args:
            patient_id: Patient identifier
            abnormality: Type of abnormality to include
            
        Returns:
            ECG reading object
        """
        if patient_id is None:
            patient_id = f"SYNTH_{random.randint(1000, 9999)}"
        
        duration = 10.0
        sampling_rate = 500
        
        # Generate signals for different leads
        if abnormality:
            lead_ii = self.generate_abnormal_ecg(duration, sampling_rate, abnormality)
        else:
            lead_ii = self.generate_ecg_signal(duration, sampling_rate)
        
        # Generate other leads with variations
        lead_i = lead_ii * 0.8 + np.random.normal(0, 0.05, len(lead_ii))
        lead_v1 = lead_ii * 0.6 + np.random.normal(0, 0.03, len(lead_ii))
        
        return ECGReading(
            timestamp=datetime.now() - timedelta(minutes=random.randint(0, 60)),
            lead_data={
                'I': lead_i.tolist(),
                'II': lead_ii.tolist(),
                'V1': lead_v1.tolist()
            },
            sampling_rate=sampling_rate,
            duration=duration,
            patient_id=patient_id,
            device_id=f"ECG_DEVICE_{random.randint(100, 999)}",
            quality_score=random.uniform(0.7, 0.95)
        )
    
    def generate_patient_vitals(self, patient_id: str = None, risk_level: str = "normal") -> PatientVitals:
        """Generate patient vitals with specified risk level.
        
        Args:
            patient_id: Patient identifier
            risk_level: Risk level ("low", "normal", "high", "critical")
            
        Returns:
            Patient vitals object
        """
        if patient_id is None:
            patient_id = f"PATIENT_{random.randint(1000, 9999)}"
        
        if risk_level == "low":
            age = random.randint(25, 45)
            gender = random.choice(["M", "F"])
            total_chol = random.uniform(150, 200)
            ldl_chol = random.uniform(70, 100)
            hdl_chol = random.uniform(50, 80)
            triglycerides = random.uniform(80, 150)
            systolic_bp = random.randint(100, 120)
            diastolic_bp = random.randint(60, 80)
            diabetes = False
            smoking = False
            previous_mi = False
            on_statins = random.choice([True, False])
        
        elif risk_level == "normal":
            age = random.randint(35, 65)
            gender = random.choice(["M", "F"])
            total_chol = random.uniform(180, 240)
            ldl_chol = random.uniform(100, 140)
            hdl_chol = random.uniform(40, 60)
            triglycerides = random.uniform(120, 200)
            systolic_bp = random.randint(110, 140)
            diastolic_bp = random.randint(70, 90)
            diabetes = random.choice([True, False])
            smoking = random.choice([True, False])
            previous_mi = False
            on_statins = random.choice([True, False])
        
        elif risk_level == "high":
            age = random.randint(55, 75)
            gender = random.choice(["M", "F"])
            total_chol = random.uniform(240, 300)
            ldl_chol = random.uniform(140, 190)
            hdl_chol = random.uniform(30, 45)
            triglycerides = random.uniform(200, 300)
            systolic_bp = random.randint(140, 170)
            diastolic_bp = random.randint(85, 100)
            diabetes = True
            smoking = True
            previous_mi = random.choice([True, False])
            on_statins = random.choice([True, False])
        
        else:  # critical
            age = random.randint(65, 85)
            gender = "M"  # Higher risk
            total_chol = random.uniform(280, 350)
            ldl_chol = random.uniform(180, 250)
            hdl_chol = random.uniform(25, 40)
            triglycerides = random.uniform(250, 400)
            systolic_bp = random.randint(160, 200)
            diastolic_bp = random.randint(95, 120)
            diabetes = True
            smoking = True
            previous_mi = True
            on_statins = False
        
        return PatientVitals(
            patient_id=patient_id,
            age=age,
            gender=gender,
            total_cholesterol=total_chol,
            ldl_cholesterol=ldl_chol,
            hdl_cholesterol=hdl_chol,
            triglycerides=triglycerides,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=random.randint(60, 100),
            diabetes=diabetes,
            smoking=smoking,
            family_history_cad=random.choice([True, False]),
            previous_mi=previous_mi,
            on_statins=on_statins,
            on_beta_blockers=random.choice([True, False]),
            on_ace_inhibitors=random.choice([True, False])
        )
    
    def generate_test_dataset(self, n_patients: int = 100) -> Dict[str, List]:
        """Generate complete test dataset.
        
        Args:
            n_patients: Number of patients to generate
            
        Returns:
            Dictionary with ECG readings and patient vitals
        """
        ecg_readings = []
        patient_vitals = []
        
        risk_levels = ["low", "normal", "high", "critical"]
        abnormalities = [None, "st_elevation", "st_depression", "tachycardia", "bradycardia", "irregular_rhythm"]
        
        for i in range(n_patients):
            patient_id = f"TEST_PATIENT_{i:04d}"
            
            # Randomly assign risk level and abnormality
            risk_level = random.choice(risk_levels)
            abnormality = random.choice(abnormalities)
            
            # Generate data
            ecg_reading = self.generate_ecg_reading(patient_id, abnormality)
            vitals = self.generate_patient_vitals(patient_id, risk_level)
            
            ecg_readings.append(ecg_reading)
            patient_vitals.append(vitals)
        
        return {
            "ecg_readings": ecg_readings,
            "patient_vitals": patient_vitals
        }


class TestSyntheticDataGenerator:
    """Test cases for synthetic data generator."""
    
    def test_generate_ecg_signal(self):
        """Test ECG signal generation."""
        generator = SyntheticDataGenerator()
        signal = generator.generate_ecg_signal(duration=5.0, sampling_rate=500)
        
        assert len(signal) == 2500  # 5 seconds * 500 Hz
        assert isinstance(signal, np.ndarray)
        assert signal.std() > 0  # Signal should have variability
    
    def test_generate_abnormal_ecg(self):
        """Test abnormal ECG generation."""
        generator = SyntheticDataGenerator()
        
        # Test different abnormalities
        abnormalities = ["st_elevation", "st_depression", "tachycardia", "bradycardia", "irregular_rhythm"]
        
        for abnormality in abnormalities:
            signal = generator.generate_abnormal_ecg(abnormality_type=abnormality)
            assert len(signal) == 5000  # 10 seconds * 500 Hz
            assert isinstance(signal, np.ndarray)
    
    def test_generate_ecg_reading(self):
        """Test ECG reading generation."""
        generator = SyntheticDataGenerator()
        reading = generator.generate_ecg_reading("TEST_001")
        
        assert reading.patient_id == "TEST_001"
        assert reading.sampling_rate == 500
        assert reading.duration == 10.0
        assert len(reading.lead_data) == 3
        assert "I" in reading.lead_data
        assert "II" in reading.lead_data
        assert "V1" in reading.lead_data
    
    def test_generate_patient_vitals(self):
        """Test patient vitals generation."""
        generator = SyntheticDataGenerator()
        
        # Test different risk levels
        for risk_level in ["low", "normal", "high", "critical"]:
            vitals = generator.generate_patient_vitals("TEST_001", risk_level)
            
            assert vitals.patient_id == "TEST_001"
            assert 18 <= vitals.age <= 100
            assert vitals.gender in ["M", "F"]
            assert vitals.total_cholesterol > 0
            assert vitals.ldl_cholesterol > 0
            assert vitals.hdl_cholesterol > 0
    
    def test_generate_test_dataset(self):
        """Test complete dataset generation."""
        generator = SyntheticDataGenerator()
        dataset = generator.generate_test_dataset(n_patients=10)
        
        assert "ecg_readings" in dataset
        assert "patient_vitals" in dataset
        assert len(dataset["ecg_readings"]) == 10
        assert len(dataset["patient_vitals"]) == 10
        
        # Check patient ID consistency
        for i in range(10):
            ecg_id = dataset["ecg_readings"][i].patient_id
            vitals_id = dataset["patient_vitals"][i].patient_id
            assert ecg_id == vitals_id
    
    def test_reproducibility(self):
        """Test that generator produces reproducible results."""
        # Generate signals with same parameters and seed
        np.random.seed(123)
        random.seed(123)
        generator1 = SyntheticDataGenerator(seed=123)
        signal1 = generator1.generate_ecg_signal()
        
        np.random.seed(123)
        random.seed(123)
        generator2 = SyntheticDataGenerator(seed=123)
        signal2 = generator2.generate_ecg_signal()
        
        np.testing.assert_array_equal(signal1, signal2)