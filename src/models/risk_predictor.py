"""Heart attack risk prediction model."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from .ecg_data import PatientVitals, ECGFeatures, RiskPrediction

logger = logging.getLogger(__name__)


class HeartAttackRiskPredictor:
    """Machine learning model for heart attack risk prediction."""
    
    def __init__(self):
        """Initialize the risk predictor."""
        self.ecg_model = None
        self.cholesterol_model = None
        self.combined_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_version = "1.0.0"
        
    def extract_cholesterol_features(self, vitals: PatientVitals) -> np.ndarray:
        """Extract features from cholesterol and vital signs data.
        
        Args:
            vitals: Patient vital signs and risk factors
            
        Returns:
            Feature array for cholesterol-based risk assessment
        """
        features = []
        
        # Basic demographics
        features.append(vitals.age)
        features.append(1 if vitals.gender == 'M' else 0)  # Male gender
        
        # Cholesterol levels (use defaults if missing)
        features.append(vitals.total_cholesterol or 200)
        features.append(vitals.ldl_cholesterol or 100)
        features.append(vitals.hdl_cholesterol or 50)
        features.append(vitals.triglycerides or 150)
        
        # Calculated ratios
        total_chol = vitals.total_cholesterol or 200
        hdl_chol = vitals.hdl_cholesterol or 50
        features.append(total_chol / hdl_chol)  # Total/HDL ratio
        
        # Blood pressure
        features.append(vitals.systolic_bp or 120)
        features.append(vitals.diastolic_bp or 80)
        features.append(vitals.heart_rate or 70)
        
        # Risk factors (binary)
        features.append(1 if vitals.diabetes else 0)
        features.append(1 if vitals.smoking else 0)
        features.append(1 if vitals.family_history_cad else 0)
        features.append(1 if vitals.previous_mi else 0)
        
        # Medications (protective factors)
        features.append(1 if vitals.on_statins else 0)
        features.append(1 if vitals.on_beta_blockers else 0)
        features.append(1 if vitals.on_ace_inhibitors else 0)
        
        return np.array(features)
    
    def extract_ecg_features(self, ecg_features: ECGFeatures) -> np.ndarray:
        """Extract features from ECG analysis.
        
        Args:
            ecg_features: Extracted ECG features
            
        Returns:
            Feature array for ECG-based risk assessment
        """
        features = []
        
        # HRV features
        features.append(ecg_features.mean_rr)
        features.append(ecg_features.sdnn)
        features.append(ecg_features.rmssd)
        features.append(ecg_features.pnn50)
        features.append(ecg_features.lf_power)
        features.append(ecg_features.hf_power)
        features.append(ecg_features.lf_hf_ratio)
        
        # Morphological features
        features.append(ecg_features.p_wave_duration or 100)
        features.append(ecg_features.pr_interval or 160)
        features.append(ecg_features.qrs_duration or 100)
        features.append(ecg_features.qt_interval or 400)
        features.append(ecg_features.qtc_interval or 420)
        
        # ST segment analysis
        st_elevation = max(ecg_features.st_elevation.values()) if ecg_features.st_elevation else 0
        st_depression = max(ecg_features.st_depression.values()) if ecg_features.st_depression else 0
        features.append(st_elevation)
        features.append(st_depression)
        
        # T wave features
        t_wave_amp = np.mean(list(ecg_features.t_wave_amplitude.values())) if ecg_features.t_wave_amplitude else 0
        t_wave_inv_count = sum(ecg_features.t_wave_inversion.values()) if ecg_features.t_wave_inversion else 0
        features.append(t_wave_amp)
        features.append(t_wave_inv_count)
        
        return np.array(features)
    
    def calculate_framingham_risk(self, vitals: PatientVitals) -> float:
        """Calculate Framingham Risk Score for 10-year cardiovascular disease risk.
        
        Args:
            vitals: Patient vital signs and risk factors
            
        Returns:
            10-year cardiovascular disease risk (0-1)
        """
        # Simplified Framingham Risk Score calculation
        points = 0
        
        # Age points
        if vitals.gender == 'M':
            if vitals.age >= 70: points += 11
            elif vitals.age >= 65: points += 10
            elif vitals.age >= 60: points += 8
            elif vitals.age >= 55: points += 6
            elif vitals.age >= 50: points += 4
            elif vitals.age >= 45: points += 2
            elif vitals.age >= 40: points += 1
        else:  # Female
            if vitals.age >= 75: points += 16
            elif vitals.age >= 70: points += 12
            elif vitals.age >= 65: points += 9
            elif vitals.age >= 60: points += 6
            elif vitals.age >= 55: points += 4
            elif vitals.age >= 50: points += 2
            elif vitals.age >= 45: points += 1
        
        # Total cholesterol points
        total_chol = vitals.total_cholesterol or 200
        if total_chol >= 280: points += 3
        elif total_chol >= 240: points += 2
        elif total_chol >= 200: points += 1
        
        # HDL cholesterol points (protective)
        hdl_chol = vitals.hdl_cholesterol or 50
        if hdl_chol < 35: points += 2
        elif hdl_chol < 45: points += 1
        elif hdl_chol >= 60: points -= 1
        
        # Blood pressure points
        systolic = vitals.systolic_bp or 120
        if systolic >= 160: points += 2
        elif systolic >= 140: points += 1
        
        # Risk factors
        if vitals.diabetes: points += 2
        if vitals.smoking: points += 2
        
        # Convert points to risk percentage (simplified)
        risk_percentage = min(0.99, max(0.01, points / 20))
        
        return risk_percentage
    
    def predict_risk(self, vitals: PatientVitals, ecg_features: ECGFeatures) -> RiskPrediction:
        """Predict heart attack risk based on patient data and ECG features.
        
        Args:
            vitals: Patient vital signs and risk factors
            ecg_features: Extracted ECG features
            
        Returns:
            Risk prediction results
        """
        # Extract features
        cholesterol_features = self.extract_cholesterol_features(vitals)
        ecg_feature_array = self.extract_ecg_features(ecg_features)
        
        # Calculate individual risk components
        cholesterol_risk = self.calculate_framingham_risk(vitals)
        
        # ECG-based risk (simplified heuristic if model not trained)
        ecg_risk = self._calculate_ecg_risk_heuristic(ecg_features)
        
        # Clinical risk factors
        clinical_risk = self._calculate_clinical_risk(vitals)
        
        # Combine risks (weighted average)
        immediate_risk = (0.4 * ecg_risk + 0.3 * cholesterol_risk + 0.3 * clinical_risk)
        short_term_risk = (0.3 * ecg_risk + 0.4 * cholesterol_risk + 0.3 * clinical_risk)
        long_term_risk = (0.2 * ecg_risk + 0.5 * cholesterol_risk + 0.3 * clinical_risk)
        
        # Determine risk category
        max_risk = max(immediate_risk, short_term_risk, long_term_risk)
        if max_risk >= 0.8:
            risk_category = "Critical"
        elif max_risk >= 0.6:
            risk_category = "High"
        elif max_risk >= 0.3:
            risk_category = "Moderate"
        else:
            risk_category = "Low"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(vitals, ecg_features, risk_category)
        
        # Determine if urgent action is required
        urgent_action = immediate_risk > 0.7 or any([
            ecg_features.st_elevation and max(ecg_features.st_elevation.values()) > 0.2,
            ecg_features.st_depression and max(ecg_features.st_depression.values()) > 0.2,
            vitals.total_cholesterol and vitals.total_cholesterol > 300,
            vitals.systolic_bp and vitals.systolic_bp > 180
        ])
        
        return RiskPrediction(
            patient_id=vitals.patient_id,
            prediction_timestamp=datetime.now(),
            immediate_risk=min(0.99, max(0.01, immediate_risk)),
            short_term_risk=min(0.99, max(0.01, short_term_risk)),
            long_term_risk=min(0.99, max(0.01, long_term_risk)),
            risk_category=risk_category,
            ecg_risk_score=min(0.99, max(0.01, ecg_risk)),
            cholesterol_risk_score=min(0.99, max(0.01, cholesterol_risk)),
            clinical_risk_score=min(0.99, max(0.01, clinical_risk)),
            recommendations=recommendations,
            urgent_action_required=urgent_action,
            model_version=self.model_version,
            confidence_score=0.85  # Default confidence
        )
    
    def _calculate_ecg_risk_heuristic(self, ecg_features: ECGFeatures) -> float:
        """Calculate ECG-based risk using heuristic rules.
        
        Args:
            ecg_features: ECG features
            
        Returns:
            ECG risk score (0-1)
        """
        risk_score = 0.0
        
        # ST segment changes (major risk factor)
        if ecg_features.st_elevation:
            max_elevation = max(ecg_features.st_elevation.values())
            risk_score += min(0.5, max_elevation * 2)
        
        if ecg_features.st_depression:
            max_depression = max(ecg_features.st_depression.values())
            risk_score += min(0.4, max_depression * 2)
        
        # T wave abnormalities
        if ecg_features.t_wave_inversion:
            inversion_count = sum(ecg_features.t_wave_inversion.values())
            risk_score += min(0.3, inversion_count * 0.1)
        
        # QT prolongation
        if ecg_features.qtc_interval and ecg_features.qtc_interval > 450:
            risk_score += min(0.2, (ecg_features.qtc_interval - 450) / 100)
        
        # HRV abnormalities
        if ecg_features.sdnn < 20:  # Low HRV
            risk_score += 0.2
        
        if ecg_features.lf_hf_ratio > 4 or ecg_features.lf_hf_ratio < 0.5:
            risk_score += 0.1
        
        return min(0.99, risk_score)
    
    def _calculate_clinical_risk(self, vitals: PatientVitals) -> float:
        """Calculate clinical risk based on patient factors.
        
        Args:
            vitals: Patient vitals
            
        Returns:
            Clinical risk score (0-1)
        """
        risk_score = 0.0
        
        # Age factor
        if vitals.age > 65:
            risk_score += 0.3
        elif vitals.age > 55:
            risk_score += 0.2
        elif vitals.age > 45:
            risk_score += 0.1
        
        # Gender factor (males higher risk)
        if vitals.gender == 'M':
            risk_score += 0.1
        
        # Previous MI
        if vitals.previous_mi:
            risk_score += 0.4
        
        # Diabetes
        if vitals.diabetes:
            risk_score += 0.2
        
        # Smoking
        if vitals.smoking:
            risk_score += 0.2
        
        # Family history
        if vitals.family_history_cad:
            risk_score += 0.1
        
        # Protective factors
        if vitals.on_statins:
            risk_score -= 0.1
        if vitals.on_beta_blockers:
            risk_score -= 0.05
        if vitals.on_ace_inhibitors:
            risk_score -= 0.05
        
        return min(0.99, max(0.01, risk_score))
    
    def _generate_recommendations(self, vitals: PatientVitals, ecg_features: ECGFeatures, 
                                risk_category: str) -> List[str]:
        """Generate personalized recommendations based on risk assessment.
        
        Args:
            vitals: Patient vitals
            ecg_features: ECG features
            risk_category: Risk category
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if risk_category == "Critical":
            recommendations.append("URGENT: Seek immediate emergency medical attention")
            recommendations.append("Call emergency services (911) immediately")
            
        elif risk_category == "High":
            recommendations.append("Schedule urgent cardiology consultation within 24-48 hours")
            recommendations.append("Consider emergency department evaluation")
            
        # Cholesterol management
        if vitals.total_cholesterol and vitals.total_cholesterol > 240:
            recommendations.append("Discuss statin therapy with physician")
            recommendations.append("Implement strict low-cholesterol diet")
            
        if vitals.ldl_cholesterol and vitals.ldl_cholesterol > 160:
            recommendations.append("Target LDL cholesterol reduction to <100 mg/dL")
            
        # Blood pressure management
        if vitals.systolic_bp and vitals.systolic_bp > 140:
            recommendations.append("Blood pressure management required")
            recommendations.append("Consider ACE inhibitor or ARB therapy")
            
        # Lifestyle modifications
        if vitals.smoking:
            recommendations.append("Smoking cessation is critical")
            recommendations.append("Consider nicotine replacement therapy")
            
        if not vitals.on_statins and (vitals.total_cholesterol or 0) > 200:
            recommendations.append("Discuss statin therapy benefits with physician")
            
        # ECG-specific recommendations
        if ecg_features.st_elevation or ecg_features.st_depression:
            recommendations.append("ECG abnormalities detected - cardiology follow-up required")
            
        if ecg_features.qtc_interval and ecg_features.qtc_interval > 450:
            recommendations.append("QT prolongation detected - medication review needed")
            
        # General recommendations
        if risk_category in ["Moderate", "High", "Critical"]:
            recommendations.extend([
                "Regular cardiovascular exercise as tolerated",
                "Mediterranean diet with omega-3 fatty acids",
                "Stress management and adequate sleep",
                "Regular blood pressure and cholesterol monitoring"
            ])
        
        return recommendations