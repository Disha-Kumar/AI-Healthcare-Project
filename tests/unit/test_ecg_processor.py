"""Unit tests for ECG processor."""

import pytest
import numpy as np
from src.utils.ecg_processor import ECGProcessor


class TestECGProcessor:
    """Test cases for ECG processor."""
    
    def test_initialization(self):
        """Test ECG processor initialization."""
        processor = ECGProcessor(sampling_rate=500)
        assert processor.sampling_rate == 500
        assert processor.nyquist_freq == 250
    
    def test_preprocess_signal(self, ecg_processor, sample_ecg_data):
        """Test signal preprocessing."""
        signal = np.array(sample_ecg_data['II'])
        processed = ecg_processor.preprocess_signal(signal)
        
        assert len(processed) == len(signal)
        assert isinstance(processed, np.ndarray)
        # Signal should be filtered (different from original)
        assert not np.array_equal(signal, processed)
    
    def test_detect_r_peaks(self, ecg_processor, sample_ecg_data):
        """Test R peak detection."""
        signal = np.array(sample_ecg_data['II'])
        processed = ecg_processor.preprocess_signal(signal)
        r_peaks = ecg_processor.detect_r_peaks(processed)
        
        assert isinstance(r_peaks, np.ndarray)
        assert len(r_peaks) > 0
        # Should detect reasonable number of peaks for 10-second signal at ~70 bpm
        assert 5 <= len(r_peaks) <= 15
    
    def test_calculate_rr_intervals(self, ecg_processor):
        """Test RR interval calculation."""
        # Mock R peaks at regular intervals (70 bpm)
        sampling_rate = 500
        rr_interval_samples = int(60 / 70 * sampling_rate)  # ~428 samples
        r_peaks = np.array([0, rr_interval_samples, 2 * rr_interval_samples, 3 * rr_interval_samples])
        
        rr_intervals = ecg_processor.calculate_rr_intervals(r_peaks)
        
        assert len(rr_intervals) == 3  # n-1 intervals for n peaks
        # Should be approximately 857ms for 70 bpm
        assert 800 <= np.mean(rr_intervals) <= 900
    
    def test_extract_hrv_features(self, ecg_processor):
        """Test HRV feature extraction."""
        # Create regular RR intervals with some variability
        base_rr = 857  # ms for 70 bpm
        rr_intervals = np.array([base_rr + np.random.normal(0, 20) for _ in range(20)])
        
        features = ecg_processor.extract_hrv_features(rr_intervals)
        
        assert 'mean_rr' in features
        assert 'sdnn' in features
        assert 'rmssd' in features
        assert 'pnn50' in features
        assert 'lf_power' in features
        assert 'hf_power' in features
        assert 'lf_hf_ratio' in features
        
        assert 800 <= features['mean_rr'] <= 900
        assert features['sdnn'] > 0
        assert features['rmssd'] >= 0
        assert 0 <= features['pnn50'] <= 100
    
    def test_extract_hrv_features_insufficient_data(self, ecg_processor):
        """Test HRV feature extraction with insufficient data."""
        rr_intervals = np.array([857, 860])  # Only 2 intervals
        
        features = ecg_processor.extract_hrv_features(rr_intervals)
        
        # Should return default values for insufficient data
        assert features['mean_rr'] == 0.0
        assert features['sdnn'] == 0.0
        assert features['rmssd'] == 0.0
        assert features['pnn50'] == 0.0
    
    def test_detect_st_changes(self, ecg_processor, sample_ecg_data):
        """Test ST segment change detection."""
        signal = np.array(sample_ecg_data['II'])
        processed = ecg_processor.preprocess_signal(signal)
        r_peaks = ecg_processor.detect_r_peaks(processed)
        
        st_changes = ecg_processor.detect_st_changes(processed, r_peaks)
        
        assert 'elevation' in st_changes
        assert 'depression' in st_changes
        assert isinstance(st_changes['elevation'], float)
        assert isinstance(st_changes['depression'], float)
        assert st_changes['elevation'] >= 0
        assert st_changes['depression'] >= 0
    
    def test_assess_signal_quality(self, ecg_processor, sample_ecg_data):
        """Test signal quality assessment."""
        signal = np.array(sample_ecg_data['II'])
        
        quality_score, noise_level = ecg_processor.assess_signal_quality(signal)
        
        assert 0 <= quality_score <= 1
        assert 0 <= noise_level <= 1
        assert isinstance(quality_score, float)
        assert isinstance(noise_level, float)
    
    def test_assess_signal_quality_noisy_signal(self, ecg_processor):
        """Test signal quality assessment with noisy signal."""
        # Create a very noisy signal
        clean_signal = np.sin(np.linspace(0, 10 * np.pi, 1000))
        noise = np.random.normal(0, 2, 1000)  # High noise
        noisy_signal = clean_signal + noise
        
        quality_score, noise_level = ecg_processor.assess_signal_quality(noisy_signal)
        
        # Noisy signal should have lower quality and higher noise level
        assert quality_score < 0.5
        assert noise_level > 0.3
    
    def test_frequency_features_edge_cases(self, ecg_processor):
        """Test frequency domain features with edge cases."""
        # Very short RR interval series
        short_rr = np.array([857, 860, 855])
        features = ecg_processor._calculate_frequency_features(short_rr)
        
        # Should return zeros for insufficient data
        assert features == (0.0, 0.0, 0.0)
        
        # Very regular RR intervals (no variability)
        regular_rr = np.array([857] * 50)
        features = ecg_processor._calculate_frequency_features(regular_rr)
        
        # Should handle regular intervals gracefully
        assert isinstance(features[0], float)
        assert isinstance(features[1], float)
        assert isinstance(features[2], float)