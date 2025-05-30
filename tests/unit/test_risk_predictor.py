"""Unit tests for risk predictor."""

import pytest
import numpy as np
from src.models.risk_predictor import HeartAttackRiskPredictor
from src.models.ecg_data import PatientVitals, ECGFeatures


class TestHeartAttackRiskPredictor:
    """Test cases for heart attack risk predictor."""
    
    def test_initialization(self):
        """Test risk predictor initialization."""
        predictor = HeartAttackRiskPredictor()
        assert predictor.model_version == "1.0.0"
        assert not predictor.is_trained
        assert predictor.ecg_model is None
        assert predictor.cholesterol_model is None
        assert predictor.combined_model is None
    
    def test_extract_cholesterol_features(self, sample_patient_vitals):
        """Test cholesterol feature extraction."""
        predictor = HeartAttackRiskPredictor()
        features = predictor.extract_cholesterol_features(sample_patient_vitals)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 17  # Expected number of features
        
        # Check specific feature values
        assert features[0] == sample_patient_vitals.age  # Age
        assert features[1] == 1  # Male gender
        assert features[2] == sample_patient_vitals.total_cholesterol
        assert features[3] == sample_patient_vitals.ldl_cholesterol
    
    def test_extract_cholesterol_features_missing_values(self):
        """Test cholesterol feature extraction with missing values."""
        predictor = HeartAttackRiskPredictor()
        
        # Create patient with minimal data
        minimal_vitals = PatientVitals(
            patient_id="MINIMAL_001",
            age=50,
            gender="F"
        )
        
        features = predictor.extract_cholesterol_features(minimal_vitals)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 17
        
        # Should use default values for missing data
        assert features[2] == 200  # Default total cholesterol
        assert features[3] == 100  # Default LDL
        assert features[4] == 50   # Default HDL
    
    def test_extract_ecg_features(self, sample_ecg_features):
        """Test ECG feature extraction."""
        predictor = HeartAttackRiskPredictor()
        features = predictor.extract_ecg_features(sample_ecg_features)
        
        assert isinstance(features, np.ndarray)
        assert len(features) == 16  # Expected number of ECG features
        
        # Check HRV features
        assert features[0] == sample_ecg_features.mean_rr
        assert features[1] == sample_ecg_features.sdnn
        assert features[2] == sample_ecg_features.rmssd
    
    def test_calculate_framingham_risk_male(self, sample_patient_vitals):
        """Test Framingham risk calculation for male patient."""
        predictor = HeartAttackRiskPredictor()
        risk = predictor.calculate_framingham_risk(sample_patient_vitals)
        
        assert 0 <= risk <= 1
        assert isinstance(risk, float)
        
        # Male, 55 years old, with risk factors should have moderate risk
        assert 0.1 <= risk <= 0.8
    
    def test_calculate_framingham_risk_female(self, low_risk_patient_vitals):
        """Test Framingham risk calculation for female patient."""
        predictor = HeartAttackRiskPredictor()
        risk = predictor.calculate_framingham_risk(low_risk_patient_vitals)
        
        assert 0 <= risk <= 1
        assert isinstance(risk, float)
        
        # Young female with good profile should have low risk
        assert risk <= 0.3
    
    def test_predict_risk_low_risk_patient(self, low_risk_patient_vitals, sample_ecg_features):
        """Test risk prediction for low-risk patient."""
        predictor = HeartAttackRiskPredictor()
        prediction = predictor.predict_risk(low_risk_patient_vitals, sample_ecg_features)
        
        assert prediction.patient_id == low_risk_patient_vitals.patient_id
        assert prediction.risk_category in ["Low", "Moderate", "High", "Critical"]
        assert 0 <= prediction.immediate_risk <= 1
        assert 0 <= prediction.short_term_risk <= 1
        assert 0 <= prediction.long_term_risk <= 1
        assert not prediction.urgent_action_required
        assert len(prediction.recommendations) > 0
    
    def test_predict_risk_high_risk_patient(self, high_risk_patient_vitals, abnormal_ecg_features):
        """Test risk prediction for high-risk patient."""
        predictor = HeartAttackRiskPredictor()
        prediction = predictor.predict_risk(high_risk_patient_vitals, abnormal_ecg_features)
        
        assert prediction.patient_id == high_risk_patient_vitals.patient_id
        assert prediction.risk_category in ["Moderate", "High", "Critical"]
        
        # High-risk patient should have elevated risk scores
        assert prediction.immediate_risk > 0.3
        assert prediction.cholesterol_risk_score > 0.3
        assert prediction.clinical_risk_score > 0.3
        
        # Should have urgent recommendations
        assert len(prediction.recommendations) > 0
        assert any("urgent" in rec.lower() or "emergency" in rec.lower() 
                  for rec in prediction.recommendations)
    
    def test_ecg_risk_heuristic_normal(self, sample_ecg_features):
        """Test ECG risk heuristic with normal features."""
        predictor = HeartAttackRiskPredictor()
        risk = predictor._calculate_ecg_risk_heuristic(sample_ecg_features)
        
        assert 0 <= risk <= 1
        assert isinstance(risk, float)
        
        # Normal ECG should have low risk
        assert risk <= 0.3
    
    def test_ecg_risk_heuristic_abnormal(self, abnormal_ecg_features):
        """Test ECG risk heuristic with abnormal features."""
        predictor = HeartAttackRiskPredictor()
        risk = predictor._calculate_ecg_risk_heuristic(abnormal_ecg_features)
        
        assert 0 <= risk <= 1
        assert isinstance(risk, float)
        
        # Abnormal ECG should have higher risk
        assert risk > 0.3
    
    def test_clinical_risk_calculation(self, high_risk_patient_vitals):
        """Test clinical risk calculation."""
        predictor = HeartAttackRiskPredictor()
        risk = predictor._calculate_clinical_risk(high_risk_patient_vitals)
        
        assert 0 <= risk <= 1
        assert isinstance(risk, float)
        
        # High-risk patient should have elevated clinical risk
        assert risk > 0.5
    
    def test_clinical_risk_with_protective_factors(self, sample_patient_vitals):
        """Test clinical risk with protective factors."""
        predictor = HeartAttackRiskPredictor()
        
        # Calculate risk without medications
        risk_without_meds = predictor._calculate_clinical_risk(sample_patient_vitals)
        
        # Add protective medications
        sample_patient_vitals.on_statins = True
        sample_patient_vitals.on_beta_blockers = True
        sample_patient_vitals.on_ace_inhibitors = True
        
        risk_with_meds = predictor._calculate_clinical_risk(sample_patient_vitals)
        
        # Risk should be lower with protective medications
        assert risk_with_meds < risk_without_meds
    
    def test_generate_recommendations_critical_risk(self, high_risk_patient_vitals, abnormal_ecg_features):
        """Test recommendation generation for critical risk."""
        predictor = HeartAttackRiskPredictor()
        recommendations = predictor._generate_recommendations(
            high_risk_patient_vitals, abnormal_ecg_features, "Critical"
        )
        
        assert len(recommendations) > 0
        assert any("URGENT" in rec or "emergency" in rec.lower() for rec in recommendations)
        assert any("911" in rec for rec in recommendations)
    
    def test_generate_recommendations_high_cholesterol(self, sample_patient_vitals, sample_ecg_features):
        """Test recommendations for high cholesterol."""
        # Set high cholesterol
        sample_patient_vitals.total_cholesterol = 280.0
        sample_patient_vitals.ldl_cholesterol = 180.0
        
        predictor = HeartAttackRiskPredictor()
        recommendations = predictor._generate_recommendations(
            sample_patient_vitals, sample_ecg_features, "Moderate"
        )
        
        assert any("statin" in rec.lower() for rec in recommendations)
        assert any("cholesterol" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_smoking(self, sample_patient_vitals, sample_ecg_features):
        """Test recommendations for smoking."""
        predictor = HeartAttackRiskPredictor()
        recommendations = predictor._generate_recommendations(
            sample_patient_vitals, sample_ecg_features, "Moderate"
        )
        
        # Patient is a smoker, should get smoking cessation recommendations
        assert any("smoking" in rec.lower() for rec in recommendations)
        assert any("cessation" in rec.lower() for rec in recommendations)
    
    def test_risk_prediction_consistency(self, sample_patient_vitals, sample_ecg_features):
        """Test that risk predictions are consistent across multiple calls."""
        predictor = HeartAttackRiskPredictor()
        
        prediction1 = predictor.predict_risk(sample_patient_vitals, sample_ecg_features)
        prediction2 = predictor.predict_risk(sample_patient_vitals, sample_ecg_features)
        
        # Results should be consistent (deterministic)
        assert prediction1.immediate_risk == prediction2.immediate_risk
        assert prediction1.short_term_risk == prediction2.short_term_risk
        assert prediction1.long_term_risk == prediction2.long_term_risk
        assert prediction1.risk_category == prediction2.risk_category
    
    def test_risk_bounds(self, sample_patient_vitals, sample_ecg_features):
        """Test that all risk scores are within valid bounds."""
        predictor = HeartAttackRiskPredictor()
        prediction = predictor.predict_risk(sample_patient_vitals, sample_ecg_features)
        
        # All risk scores should be between 0 and 1
        assert 0 <= prediction.immediate_risk <= 1
        assert 0 <= prediction.short_term_risk <= 1
        assert 0 <= prediction.long_term_risk <= 1
        assert 0 <= prediction.ecg_risk_score <= 1
        assert 0 <= prediction.cholesterol_risk_score <= 1
        assert 0 <= prediction.clinical_risk_score <= 1
        assert 0 <= prediction.confidence_score <= 1