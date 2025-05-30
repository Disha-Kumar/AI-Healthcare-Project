"""ECG signal processing utilities."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import zscore
import logging

logger = logging.getLogger(__name__)


class ECGProcessor:
    """ECG signal processing and feature extraction."""
    
    def __init__(self, sampling_rate: int = 500):
        """Initialize ECG processor.
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
        
    def preprocess_signal(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Preprocess ECG signal with filtering and noise removal.
        
        Args:
            ecg_signal: Raw ECG signal
            
        Returns:
            Preprocessed ECG signal
        """
        # Remove baseline wander (high-pass filter at 0.5 Hz)
        sos_hp = signal.butter(4, 0.5 / self.nyquist_freq, btype='high', output='sos')
        filtered_signal = signal.sosfilt(sos_hp, ecg_signal)
        
        # Remove high-frequency noise (low-pass filter at 40 Hz)
        sos_lp = signal.butter(4, 40 / self.nyquist_freq, btype='low', output='sos')
        filtered_signal = signal.sosfilt(sos_lp, filtered_signal)
        
        # Notch filter for power line interference (50/60 Hz)
        for freq in [50, 60]:
            if freq < self.nyquist_freq:
                b_notch, a_notch = signal.iirnotch(freq, 30, self.sampling_rate)
                filtered_signal = signal.filtfilt(b_notch, a_notch, filtered_signal)
        
        return filtered_signal
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Detect R peaks in ECG signal.
        
        Args:
            ecg_signal: Preprocessed ECG signal
            
        Returns:
            Array of R peak indices
        """
        # Simple R peak detection using scipy
        peaks, properties = signal.find_peaks(
            ecg_signal,
            height=np.std(ecg_signal) * 0.5,
            distance=int(0.6 * self.sampling_rate)  # Minimum 600ms between peaks
        )
        
        # Filter peaks based on prominence
        prominences = signal.peak_prominences(ecg_signal, peaks)[0]
        valid_peaks = peaks[prominences > np.percentile(prominences, 50)]
        
        return valid_peaks
    
    def calculate_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """Calculate RR intervals from R peaks.
        
        Args:
            r_peaks: Array of R peak indices
            
        Returns:
            RR intervals in milliseconds
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        rr_intervals = np.diff(r_peaks) / self.sampling_rate * 1000  # Convert to ms
        return rr_intervals
    
    def extract_hrv_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Extract heart rate variability features.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            
        Returns:
            Dictionary of HRV features
        """
        if len(rr_intervals) < 5:
            return {
                'mean_rr': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'pnn50': 0.0,
                'lf_power': 0.0,
                'hf_power': 0.0,
                'lf_hf_ratio': 0.0
            }
        
        # Time domain features
        mean_rr = np.mean(rr_intervals)
        sdnn = np.std(rr_intervals)
        
        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        # pNN50: Percentage of successive RR intervals that differ by more than 50ms
        pnn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100
        
        # Frequency domain features
        lf_power, hf_power, lf_hf_ratio = self._calculate_frequency_features(rr_intervals)
        
        return {
            'mean_rr': float(mean_rr),
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'pnn50': float(pnn50),
            'lf_power': float(lf_power),
            'hf_power': float(hf_power),
            'lf_hf_ratio': float(lf_hf_ratio)
        }
    
    def _calculate_frequency_features(self, rr_intervals: np.ndarray) -> Tuple[float, float, float]:
        """Calculate frequency domain HRV features.
        
        Args:
            rr_intervals: RR intervals in milliseconds
            
        Returns:
            Tuple of (LF power, HF power, LF/HF ratio)
        """
        if len(rr_intervals) < 10:
            return 0.0, 0.0, 0.0
        
        # Interpolate RR intervals to regular time series
        time_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
        time_interp = np.arange(0, time_rr[-1], 1/4)  # 4 Hz interpolation
        
        if len(time_interp) < 10:
            return 0.0, 0.0, 0.0
        
        rr_interp = np.interp(time_interp, time_rr[:-1], rr_intervals[:-1])
        
        # Calculate power spectral density
        freqs, psd = signal.welch(rr_interp, fs=4, nperseg=min(256, len(rr_interp)))
        
        # Define frequency bands
        lf_band = (freqs >= 0.04) & (freqs <= 0.15)
        hf_band = (freqs >= 0.15) & (freqs <= 0.4)
        
        lf_power = np.trapz(psd[lf_band], freqs[lf_band])
        hf_power = np.trapz(psd[hf_band], freqs[hf_band])
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
        
        return lf_power, hf_power, lf_hf_ratio
    
    def detect_st_changes(self, ecg_signal: np.ndarray, r_peaks: np.ndarray) -> Dict[str, float]:
        """Detect ST segment elevation/depression.
        
        Args:
            ecg_signal: Preprocessed ECG signal
            r_peaks: R peak locations
            
        Returns:
            Dictionary with ST elevation/depression measurements
        """
        st_measurements = {'elevation': 0.0, 'depression': 0.0}
        
        if len(r_peaks) < 2:
            return st_measurements
        
        # Analyze ST segment (typically 80ms after R peak)
        st_offset = int(0.08 * self.sampling_rate)  # 80ms
        
        st_values = []
        for r_peak in r_peaks:
            if r_peak + st_offset < len(ecg_signal):
                st_point = ecg_signal[r_peak + st_offset]
                baseline = np.mean(ecg_signal[max(0, r_peak - 50):r_peak - 10])
                st_deviation = st_point - baseline
                st_values.append(st_deviation)
        
        if st_values:
            mean_st = np.mean(st_values)
            if mean_st > 0.1:  # Threshold for elevation (mV)
                st_measurements['elevation'] = float(mean_st)
            elif mean_st < -0.1:  # Threshold for depression (mV)
                st_measurements['depression'] = float(abs(mean_st))
        
        return st_measurements
    
    def assess_signal_quality(self, ecg_signal: np.ndarray) -> Tuple[float, float]:
        """Assess ECG signal quality.
        
        Args:
            ecg_signal: ECG signal
            
        Returns:
            Tuple of (quality_score, noise_level)
        """
        # Calculate signal-to-noise ratio
        signal_power = np.var(ecg_signal)
        
        # Estimate noise using high-frequency components
        sos_hp = signal.butter(4, 20 / self.nyquist_freq, btype='high', output='sos')
        noise_signal = signal.sosfilt(sos_hp, ecg_signal)
        noise_power = np.var(noise_signal)
        
        snr = signal_power / (noise_power + 1e-10)
        
        # Quality score based on SNR and signal characteristics
        quality_score = min(1.0, snr / 100)  # Normalize to 0-1
        noise_level = min(1.0, noise_power / signal_power)
        
        return float(quality_score), float(noise_level)