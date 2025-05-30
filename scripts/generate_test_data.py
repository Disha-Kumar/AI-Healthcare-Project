#!/usr/bin/env python3
"""Script to generate synthetic test data for ECG analysis."""

import json
import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.data.test_synthetic_data import SyntheticDataGenerator


def save_dataset(dataset, output_dir: Path, format_type: str = "json"):
    """Save dataset to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format_type == "json":
        # Save ECG readings
        ecg_file = output_dir / "ecg_readings.json"
        ecg_data = []
        for reading in dataset["ecg_readings"]:
            ecg_data.append({
                "timestamp": reading.timestamp.isoformat(),
                "lead_data": reading.lead_data,
                "sampling_rate": reading.sampling_rate,
                "duration": reading.duration,
                "patient_id": reading.patient_id,
                "device_id": reading.device_id,
                "quality_score": reading.quality_score
            })
        
        with open(ecg_file, 'w') as f:
            json.dump(ecg_data, f, indent=2)
        
        # Save patient vitals
        vitals_file = output_dir / "patient_vitals.json"
        vitals_data = []
        for vitals in dataset["patient_vitals"]:
            vitals_data.append({
                "patient_id": vitals.patient_id,
                "age": vitals.age,
                "gender": vitals.gender,
                "total_cholesterol": vitals.total_cholesterol,
                "ldl_cholesterol": vitals.ldl_cholesterol,
                "hdl_cholesterol": vitals.hdl_cholesterol,
                "triglycerides": vitals.triglycerides,
                "systolic_bp": vitals.systolic_bp,
                "diastolic_bp": vitals.diastolic_bp,
                "heart_rate": vitals.heart_rate,
                "diabetes": vitals.diabetes,
                "smoking": vitals.smoking,
                "family_history_cad": vitals.family_history_cad,
                "previous_mi": vitals.previous_mi,
                "on_statins": vitals.on_statins,
                "on_beta_blockers": vitals.on_beta_blockers,
                "on_ace_inhibitors": vitals.on_ace_inhibitors
            })
        
        with open(vitals_file, 'w') as f:
            json.dump(vitals_data, f, indent=2)
        
        print(f"✅ Saved {len(ecg_data)} ECG readings to {ecg_file}")
        print(f"✅ Saved {len(vitals_data)} patient vitals to {vitals_file}")


def main():
    """Generate synthetic test data."""
    parser = argparse.ArgumentParser(description="Generate synthetic ECG test data")
    parser.add_argument("--patients", "-n", type=int, default=50, 
                       help="Number of patients to generate")
    parser.add_argument("--output", "-o", type=str, default="data/synthetic",
                       help="Output directory")
    parser.add_argument("--format", choices=["json"], default="json",
                       help="Output format")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--abnormal-ratio", type=float, default=0.3,
                       help="Ratio of patients with abnormalities")
    parser.add_argument("--high-risk-ratio", type=float, default=0.2,
                       help="Ratio of high-risk patients")
    
    args = parser.parse_args()
    
    # Change to project root directory
    os.chdir(project_root)
    
    print(f"Generating synthetic ECG data...")
    print(f"Patients: {args.patients}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format}")
    print(f"Seed: {args.seed}")
    print(f"Abnormal ratio: {args.abnormal_ratio}")
    print(f"High-risk ratio: {args.high_risk_ratio}")
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate dataset
    print("\n🔄 Generating dataset...")
    dataset = generator.generate_test_dataset(n_patients=args.patients)
    
    # Save dataset
    output_dir = Path(args.output)
    save_dataset(dataset, output_dir, args.format)
    
    # Generate summary statistics
    print(f"\n📊 Dataset Summary:")
    print(f"Total patients: {len(dataset['patient_vitals'])}")
    
    # Analyze risk distribution
    risk_counts = {"low": 0, "normal": 0, "high": 0, "critical": 0}
    age_groups = {"<40": 0, "40-60": 0, ">60": 0}
    gender_counts = {"M": 0, "F": 0}
    
    for vitals in dataset["patient_vitals"]:
        # Age groups
        if vitals.age < 40:
            age_groups["<40"] += 1
        elif vitals.age <= 60:
            age_groups["40-60"] += 1
        else:
            age_groups[">60"] += 1
        
        # Gender
        gender_counts[vitals.gender] += 1
    
    print(f"\nAge distribution:")
    for age_group, count in age_groups.items():
        print(f"  {age_group}: {count} ({count/args.patients*100:.1f}%)")
    
    print(f"\nGender distribution:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count} ({count/args.patients*100:.1f}%)")
    
    # Save metadata
    metadata = {
        "generation_timestamp": datetime.now().isoformat(),
        "generator_version": "1.0.0",
        "parameters": {
            "n_patients": args.patients,
            "seed": args.seed,
            "abnormal_ratio": args.abnormal_ratio,
            "high_risk_ratio": args.high_risk_ratio
        },
        "statistics": {
            "total_patients": len(dataset["patient_vitals"]),
            "age_distribution": age_groups,
            "gender_distribution": gender_counts
        }
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to {metadata_file}")
    print(f"\n🎉 Dataset generation completed successfully!")


if __name__ == "__main__":
    main()