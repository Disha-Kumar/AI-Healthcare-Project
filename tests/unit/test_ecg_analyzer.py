"""Unit tests for ECG analyzer."""

import pytest
from datetime import datetime
from src.models.ecg_analyzer import ECGAnalyzer
from src.models.ecg_data import ECGAnalysisResult, ECGReading


class TestECGAnalyzer:
    """Test cases for ECG analyzer."""
    
    def test_initialization(self):
        """Test ECG analyzer initialization."""
        analyzer = ECGAnalyzer(sampling_rate=500)
        assert analyzer.processor.sampling_rate == 500
        assert analyzer.risk_predictor is not None
        assert analyzer.algorithm_version == "1.0.0"
    
    def test_analyze_ecg_basic(self, ecg_analyzer, sample_ecg_reading):
        """Test basic ECG analysis without patient vitals."""
        result = ecg_analyzer.analyze_ecg(sample_ecg_reading)
        
        assert isinstance(result, ECGAnalysisResult)
        assert result.patient_id == sample_ecg_reading.patient_id
        assert result.ecg_reading_id.startswith(sample_ecg_reading.patient_id)
        assert isinstance(result.analysis_timestamp, datetime)
        
        # Signal quality assessment
        assert 0 <= result.signal_quality <= 1
        assert 0 <= result.noise_level <= 1
        
        # Features should be extracted
        assert result.features is not None
        assert result.features.mean_rr > 0
        assert result.features.sdnn >= 0
        
        # Processing metadata
        assert result.processing_time_ms > 0
        assert result.algorithm_version == "1.0.0"
        
        # No risk prediction without patient vitals
        assert result.risk_prediction is None
    
    def test_analyze_ecg_with_vitals(self, ecg_analyzer, sample_ecg_reading, sample_patient_vitals):
        """Test ECG analysis with patient vitals."""
        result = ecg_analyzer.analyze_ecg(sample_ecg_reading, sample_patient_vitals)
        
        assert isinstance(result, ECGAnalysisResult)
        assert result.risk_prediction is not None
        assert result.risk_prediction.patient_id == sample_patient_vitals.patient_id
        
        # Risk scores should be present
        assert 0 <= result.risk_prediction.immediate_risk <= 1
        assert 0 <= result.risk_prediction.short_term_risk <= 1
        assert 0 <= result.risk_prediction.long_term_risk <= 1
        
        # Recommendations should be generated
        assert len(result.risk_prediction.recommendations) > 0
    
    def test_analyze_ecg_high_risk_patient(self, ecg_analyzer, sample_ecg_reading, high_risk_patient_vitals):
        """Test ECG analysis with high-risk patient."""
        # Update patient ID to match
        sample_ecg_reading.patient_id = high_risk_patient_vitals.patient_id
        
        result = ecg_analyzer.analyze_ecg(sample_ecg_reading, high_risk_patient_vitals)
        
        assert result.risk_prediction is not None
        assert result.risk_prediction.risk_category in ["Moderate", "High", "Critical"]
        
        # High-risk patient should have elevated risk scores
        assert result.risk_prediction.immediate_risk > 0.2
        assert result.risk_prediction.cholesterol_risk_score > 0.2
        
        # Should have comprehensive recommendations
        assert len(result.risk_prediction.recommendations) >= 3
    
    def test_detect_rhythm_abnormalities_normal(self, ecg_analyzer):
        """Test rhythm abnormality detection with normal HRV."""
        normal_hrv = {
            'mean_rr': 857.0,  # ~70 bpm
            'sdnn': 45.0,
            'rmssd': 35.0,
            'pnn50': 15.0,
            'lf_power': 500.0,
            'hf_power': 300.0,
            'lf_hf_ratio': 1.67
        }
        
        import numpy as np
        rr_intervals = np.array([850, 860, 855, 865, 850, 870])
        
        abnormalities = ecg_analyzer._detect_rhythm_abnormalities(normal_hrv, rr_intervals)
        
        # Normal rhythm should have minimal abnormalities
        assert len(abnormalities) <= 1
    
    def test_detect_rhythm_abnormalities_bradycardia(self, ecg_analyzer):
        """Test detection of bradycardia."""
        bradycardia_hrv = {
            'mean_rr': 1200.0,  # ~50 bpm
            'sdnn': 30.0,
            'rmssd': 25.0,
            'pnn50': 10.0,
            'lf_power': 300.0,
            'hf_power': 200.0,
            'lf_hf_ratio': 1.5
        }
        
        import numpy as np
        rr_intervals = np.array([1200, 1180, 1220, 1190, 1210])
        
        abnormalities = ecg_analyzer._detect_rhythm_abnormalities(bradycardia_hrv, rr_intervals)
        
        assert "Bradycardia" in abnormalities
    
    def test_detect_rhythm_abnormalities_tachycardia(self, ecg_analyzer):
        """Test detection of tachycardia."""
        tachycardia_hrv = {
            'mean_rr': 500.0,  # ~120 bpm
            'sdnn': 20.0,
            'rmssd': 15.0,
            'pnn50': 5.0,
            'lf_power': 200.0,
            'hf_power': 100.0,
            'lf_hf_ratio': 2.0
        }
        
        import numpy as np
        rr_intervals = np.array([500, 480, 520, 490, 510])
        
        abnormalities = ecg_analyzer._detect_rhythm_abnormalities(tachycardia_hrv, rr_intervals)
        
        assert "Tachycardia" in abnormalities
    
    def test_detect_rhythm_abnormalities_irregular(self, ecg_analyzer):
        """Test detection of irregular rhythm."""
        irregular_hrv = {
            'mean_rr': 857.0,
            'sdnn': 120.0,  # High variability
            'rmssd': 80.0,
            'pnn50': 25.0,
            'lf_power': 800.0,
            'hf_power': 400.0,
            'lf_hf_ratio': 2.0
        }
        
        import numpy as np
        rr_intervals = np.array([800, 950, 750, 900, 820, 880])
        
        abnormalities = ecg_analyzer._detect_rhythm_abnormalities(irregular_hrv, rr_intervals)
        
        assert "Irregular rhythm" in abnormalities
    
    def test_detect_morphology_abnormalities_normal(self, ecg_analyzer):
        """Test morphology abnormality detection with normal features."""
        normal_morphology = {
            'p_wave_duration': 100.0,
            'pr_interval': 160.0,
            'qrs_duration': 100.0,
            'qt_interval': 400.0,
            'qtc_interval': 420.0,
            't_wave_amplitude': 0.3,
            't_wave_inversion': False
        }
        
        normal_st = {'elevation': 0.0, 'depression': 0.0}
        
        abnormalities = ecg_analyzer._detect_morphology_abnormalities(normal_morphology, normal_st)
        
        # Normal morphology should have no abnormalities
        assert len(abnormalities) == 0
    
    def test_detect_morphology_abnormalities_qt_prolongation(self, ecg_analyzer):
        """Test detection of QT prolongation."""
        prolonged_qt = {
            'p_wave_duration': 100.0,
            'pr_interval': 160.0,
            'qrs_duration': 100.0,
            'qt_interval': 480.0,
            'qtc_interval': 520.0,  # Prolonged
            't_wave_amplitude': 0.3,
            't_wave_inversion': False
        }
        
        normal_st = {'elevation': 0.0, 'depression': 0.0}
        
        abnormalities = ecg_analyzer._detect_morphology_abnormalities(prolonged_qt, normal_st)
        
        assert "QT prolongation" in abnormalities
    
    def test_detect_morphology_abnormalities_wide_qrs(self, ecg_analyzer):
        """Test detection of wide QRS complex."""
        wide_qrs = {
            'p_wave_duration': 100.0,
            'pr_interval': 160.0,
            'qrs_duration': 140.0,  # Wide
            'qt_interval': 400.0,
            'qtc_interval': 420.0,
            't_wave_amplitude': 0.3,
            't_wave_inversion': False
        }
        
        normal_st = {'elevation': 0.0, 'depression': 0.0}
        
        abnormalities = ecg_analyzer._detect_morphology_abnormalities(wide_qrs, normal_st)
        
        assert "Wide QRS complex" in abnormalities
    
    def test_detect_morphology_abnormalities_st_elevation(self, ecg_analyzer):
        """Test detection of ST elevation."""
        normal_morphology = {
            'p_wave_duration': 100.0,
            'pr_interval': 160.0,
            'qrs_duration': 100.0,
            'qt_interval': 400.0,
            'qtc_interval': 420.0,
            't_wave_amplitude': 0.3,
            't_wave_inversion': False
        }
        
        st_elevation = {'elevation': 0.25, 'depression': 0.0}  # Significant elevation
        
        abnormalities = ecg_analyzer._detect_morphology_abnormalities(normal_morphology, st_elevation)
        
        assert "ST elevation" in abnormalities
    
    def test_detect_morphology_abnormalities_t_wave_inversion(self, ecg_analyzer):
        """Test detection of T wave inversion."""
        t_wave_abnormal = {
            'p_wave_duration': 100.0,
            'pr_interval': 160.0,
            'qrs_duration': 100.0,
            'qt_interval': 400.0,
            'qtc_interval': 420.0,
            't_wave_amplitude': 0.1,
            't_wave_inversion': True  # Inverted
        }
        
        normal_st = {'elevation': 0.0, 'depression': 0.0}
        
        abnormalities = ecg_analyzer._detect_morphology_abnormalities(t_wave_abnormal, normal_st)
        
        assert "T wave inversion" in abnormalities
    
    def test_batch_analyze(self, ecg_analyzer, sample_ecg_reading, sample_patient_vitals):
        """Test batch analysis of multiple ECG readings."""
        # Create multiple ECG readings
        ecg_readings = []
        patient_vitals = {}
        
        for i in range(3):
            reading = ECGReading(
                patient_id=f"BATCH_TEST_{i:03d}",
                timestamp=sample_ecg_reading.timestamp,
                lead_data=sample_ecg_reading.lead_data.copy(),
                sampling_rate=sample_ecg_reading.sampling_rate,
                duration=sample_ecg_reading.duration,
                device_id=sample_ecg_reading.device_id,
                quality_score=sample_ecg_reading.quality_score
            )
            ecg_readings.append(reading)
            
            vitals = sample_patient_vitals.model_copy()
            vitals.patient_id = f"BATCH_TEST_{i:03d}"
            patient_vitals[vitals.patient_id] = vitals
        
        results = ecg_analyzer.batch_analyze(ecg_readings, patient_vitals)
        
        assert len(results) == 3
        
        for i, result in enumerate(results):
            assert result.patient_id == f"BATCH_TEST_{i:03d}"
            assert result.risk_prediction is not None
            assert isinstance(result, ECGAnalysisResult)
    
    def test_batch_analyze_without_vitals(self, ecg_analyzer, sample_ecg_reading):
        """Test batch analysis without patient vitals."""
        # Create multiple ECG readings
        ecg_readings = []
        
        for i in range(2):
            reading = ECGReading(
                patient_id=f"NO_VITALS_{i:03d}",
                timestamp=sample_ecg_reading.timestamp,
                lead_data=sample_ecg_reading.lead_data.copy(),
                sampling_rate=sample_ecg_reading.sampling_rate,
                duration=sample_ecg_reading.duration,
                device_id=sample_ecg_reading.device_id,
                quality_score=sample_ecg_reading.quality_score
            )
            ecg_readings.append(reading)
        
        results = ecg_analyzer.batch_analyze(ecg_readings)
        
        assert len(results) == 2
        
        for result in results:
            assert result.risk_prediction is None  # No vitals provided
            assert isinstance(result, ECGAnalysisResult)
    
    def test_extract_morphological_features_insufficient_peaks(self, ecg_analyzer):
        """Test morphological feature extraction with insufficient R peaks."""
        import numpy as np
        signal = np.random.normal(0, 0.1, 1000)
        r_peaks = np.array([100])  # Only one peak
        
        features = ecg_analyzer._extract_morphological_features(signal, r_peaks)
        
        # Should return default values
        assert features['p_wave_duration'] == 100.0
        assert features['pr_interval'] == 160.0
        assert features['qrs_duration'] == 100.0
        assert features['qt_interval'] == 400.0
        assert features['qtc_interval'] == 420.0