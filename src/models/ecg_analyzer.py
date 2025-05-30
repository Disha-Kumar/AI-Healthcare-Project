"""Main ECG analysis engine."""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import time

from .ecg_data import ECGReading, PatientVitals, ECGFeatures, ECGAnalysisResult, RiskPrediction
from .risk_predictor import HeartAttackRiskPredictor
from ..utils.ecg_processor import ECGProcessor

logger = logging.getLogger(__name__)


class ECGAnalyzer:
    """Main ECG analysis engine that coordinates processing and risk prediction."""
    
    def __init__(self, sampling_rate: int = 500):
        """Initialize ECG analyzer.
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.processor = ECGProcessor(sampling_rate)
        self.risk_predictor = HeartAttackRiskPredictor()
        self.algorithm_version = "1.0.0"
        
    def analyze_ecg(self, ecg_reading: ECGReading, patient_vitals: Optional[PatientVitals] = None) -> ECGAnalysisResult:
        """Perform complete ECG analysis including risk prediction.
        
        Args:
            ecg_reading: ECG reading data
            patient_vitals: Optional patient vital signs and risk factors
            
        Returns:
            Complete ECG analysis result
        """
        start_time = time.time()
        
        try:
            # Process each lead
            lead_features = {}
            all_abnormalities = {
                'rhythm': [],
                'morphology': []
            }
            
            # Use lead II for primary analysis (most common for rhythm analysis)
            primary_lead = 'II'
            if primary_lead not in ecg_reading.lead_data:
                primary_lead = list(ecg_reading.lead_data.keys())[0]
            
            primary_signal = np.array(ecg_reading.lead_data[primary_lead])
            
            # Preprocess signal
            processed_signal = self.processor.preprocess_signal(primary_signal)
            
            # Assess signal quality
            quality_score, noise_level = self.processor.assess_signal_quality(processed_signal)
            
            # Detect R peaks
            r_peaks = self.processor.detect_r_peaks(processed_signal)
            
            # Calculate RR intervals
            rr_intervals = self.processor.calculate_rr_intervals(r_peaks)
            
            # Extract HRV features
            hrv_features = self.processor.extract_hrv_features(rr_intervals)
            
            # Detect ST changes
            st_changes = self.processor.detect_st_changes(processed_signal, r_peaks)
            
            # Extract morphological features (simplified)
            morphological_features = self._extract_morphological_features(processed_signal, r_peaks)
            
            # Detect abnormalities
            rhythm_abnormalities = self._detect_rhythm_abnormalities(hrv_features, rr_intervals)
            morphology_abnormalities = self._detect_morphology_abnormalities(morphological_features, st_changes)
            
            # Create ECG features object
            ecg_features = ECGFeatures(
                mean_rr=hrv_features['mean_rr'],
                sdnn=hrv_features['sdnn'],
                rmssd=hrv_features['rmssd'],
                pnn50=hrv_features['pnn50'],
                lf_power=hrv_features['lf_power'],
                hf_power=hrv_features['hf_power'],
                lf_hf_ratio=hrv_features['lf_hf_ratio'],
                p_wave_duration=morphological_features.get('p_wave_duration'),
                pr_interval=morphological_features.get('pr_interval'),
                qrs_duration=morphological_features.get('qrs_duration'),
                qt_interval=morphological_features.get('qt_interval'),
                qtc_interval=morphological_features.get('qtc_interval'),
                st_elevation={primary_lead: st_changes.get('elevation', 0.0)},
                st_depression={primary_lead: st_changes.get('depression', 0.0)},
                t_wave_amplitude={primary_lead: morphological_features.get('t_wave_amplitude', 0.0)},
                t_wave_inversion={primary_lead: morphological_features.get('t_wave_inversion', False)}
            )
            
            # Predict risk if patient vitals are available
            risk_prediction = None
            if patient_vitals:
                risk_prediction = self.risk_predictor.predict_risk(patient_vitals, ecg_features)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return ECGAnalysisResult(
                patient_id=ecg_reading.patient_id,
                analysis_timestamp=datetime.now(),
                ecg_reading_id=f"{ecg_reading.patient_id}_{ecg_reading.timestamp.isoformat()}",
                signal_quality=quality_score,
                noise_level=noise_level,
                features=ecg_features,
                rhythm_abnormalities=rhythm_abnormalities,
                morphology_abnormalities=morphology_abnormalities,
                risk_prediction=risk_prediction,
                processing_time_ms=processing_time,
                algorithm_version=self.algorithm_version
            )
            
        except Exception as e:
            logger.error(f"Error analyzing ECG: {str(e)}")
            raise
    
    def _extract_morphological_features(self, signal: np.ndarray, r_peaks: np.ndarray) -> Dict[str, float]:
        """Extract morphological features from ECG signal.
        
        Args:
            signal: Preprocessed ECG signal
            r_peaks: R peak locations
            
        Returns:
            Dictionary of morphological features
        """
        features = {}
        
        if len(r_peaks) < 2:
            return {
                'p_wave_duration': 100.0,
                'pr_interval': 160.0,
                'qrs_duration': 100.0,
                'qt_interval': 400.0,
                'qtc_interval': 420.0,
                't_wave_amplitude': 0.0,
                't_wave_inversion': False
            }
        
        # Simplified morphological analysis
        sampling_rate = self.processor.sampling_rate
        
        # Average RR interval for calculations
        rr_intervals = np.diff(r_peaks) / sampling_rate * 1000  # ms
        mean_rr = np.mean(rr_intervals)
        
        # Estimate intervals (simplified)
        features['p_wave_duration'] = 100.0  # Default P wave duration
        features['pr_interval'] = 160.0      # Default PR interval
        features['qrs_duration'] = 100.0     # Default QRS duration
        features['qt_interval'] = 400.0      # Default QT interval
        
        # QT correction (Bazett's formula)
        qt_interval = features['qt_interval']
        rr_seconds = mean_rr / 1000
        qtc = qt_interval / np.sqrt(rr_seconds)
        features['qtc_interval'] = qtc
        
        # T wave analysis (simplified)
        t_wave_amplitudes = []
        for r_peak in r_peaks:
            # Look for T wave approximately 200-400ms after R peak
            t_start = r_peak + int(0.2 * sampling_rate)
            t_end = r_peak + int(0.4 * sampling_rate)
            
            if t_end < len(signal):
                t_segment = signal[t_start:t_end]
                if len(t_segment) > 0:
                    t_wave_amplitudes.append(np.max(t_segment) - np.min(t_segment))
        
        if t_wave_amplitudes:
            features['t_wave_amplitude'] = np.mean(t_wave_amplitudes)
            # Simple T wave inversion detection
            features['t_wave_inversion'] = features['t_wave_amplitude'] < 0.1
        else:
            features['t_wave_amplitude'] = 0.0
            features['t_wave_inversion'] = False
        
        return features
    
    def _detect_rhythm_abnormalities(self, hrv_features: Dict[str, float], rr_intervals: np.ndarray) -> List[str]:
        """Detect rhythm abnormalities.
        
        Args:
            hrv_features: HRV features
            rr_intervals: RR intervals in milliseconds
            
        Returns:
            List of detected rhythm abnormalities
        """
        abnormalities = []
        
        if len(rr_intervals) < 5:
            return abnormalities
        
        # Calculate heart rate
        mean_hr = 60000 / hrv_features['mean_rr']  # Convert from RR interval to HR
        
        # Bradycardia
        if mean_hr < 60:
            abnormalities.append("Bradycardia")
        
        # Tachycardia
        if mean_hr > 100:
            abnormalities.append("Tachycardia")
        
        # Irregular rhythm (high variability)
        if hrv_features['sdnn'] > 100:
            abnormalities.append("Irregular rhythm")
        
        # Atrial fibrillation indicators
        if hrv_features['sdnn'] > 150 and hrv_features['pnn50'] > 20:
            abnormalities.append("Possible atrial fibrillation")
        
        # Very low HRV (autonomic dysfunction)
        if hrv_features['sdnn'] < 20:
            abnormalities.append("Reduced heart rate variability")
        
        return abnormalities
    
    def _detect_morphology_abnormalities(self, morphological_features: Dict[str, float], 
                                       st_changes: Dict[str, float]) -> List[str]:
        """Detect morphological abnormalities.
        
        Args:
            morphological_features: Morphological features
            st_changes: ST segment changes
            
        Returns:
            List of detected morphological abnormalities
        """
        abnormalities = []
        
        # QT prolongation
        if morphological_features.get('qtc_interval', 0) > 450:
            abnormalities.append("QT prolongation")
        
        # QRS widening
        if morphological_features.get('qrs_duration', 0) > 120:
            abnormalities.append("Wide QRS complex")
        
        # ST elevation
        if st_changes.get('elevation', 0) > 0.1:
            abnormalities.append("ST elevation")
        
        # ST depression
        if st_changes.get('depression', 0) > 0.1:
            abnormalities.append("ST depression")
        
        # T wave abnormalities
        if morphological_features.get('t_wave_inversion', False):
            abnormalities.append("T wave inversion")
        
        if morphological_features.get('t_wave_amplitude', 0) > 1.0:
            abnormalities.append("Tall T waves")
        
        return abnormalities
    
    def batch_analyze(self, ecg_readings: List[ECGReading], 
                     patient_vitals: Optional[Dict[str, PatientVitals]] = None) -> List[ECGAnalysisResult]:
        """Analyze multiple ECG readings in batch.
        
        Args:
            ecg_readings: List of ECG readings
            patient_vitals: Optional dictionary mapping patient IDs to vitals
            
        Returns:
            List of analysis results
        """
        results = []
        
        for ecg_reading in ecg_readings:
            try:
                vitals = None
                if patient_vitals and ecg_reading.patient_id in patient_vitals:
                    vitals = patient_vitals[ecg_reading.patient_id]
                
                result = self.analyze_ecg(ecg_reading, vitals)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing ECG for patient {ecg_reading.patient_id}: {str(e)}")
                continue
        
        return results