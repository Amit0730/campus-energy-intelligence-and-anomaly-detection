"""
Training & Evaluation Execution Script
Runs the end-to-end Machine Learning pipeline for Project 15:
Campus Energy Consumption Intelligence & Anomaly Detection.
"""

import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import CampusEnergyPipeline

def main():
    print("=================================================================")
    print("   INT395: SUPERVISED LEARNING - CONTINUOUS ASSESSMENT PROJECT   ")
    print("   PROJECT 15: CAMPUS ENERGY CONSUMPTION & ANOMALY DETECTION    ")
    print("=================================================================")
    
    pipeline = CampusEnergyPipeline(data_dir="data", model_dir="saved_models")
    metrics = pipeline.run_full_pipeline(num_days=730, seed=42)
    
    print("\n=================================================================")
    print("   FINAL PIPELINE SUMMARY & VALIDATION METRICS                  ")
    print("=================================================================")
    print(f"Top Forecasting Model: {metrics['best_model']}")
    print(f"Total Feature Count: {metrics['feature_count']}")
    print(f"Total Operational Hours Processed: {metrics['num_records_total']}")
    print(f"Test Set Evaluation Hours: {metrics['num_test_records']}")
    print(f"Anomaly Detection Precision: {metrics['anomaly_metrics']['Precision']:.4f}")
    print(f"Anomaly Detection Recall: {metrics['anomaly_metrics']['Recall']:.4f}")
    print(f"Anomaly Detection F1 Score: {metrics['anomaly_metrics']['F1_Score']:.4f}")
    print("\nTraining completed successfully. Ready for Streamlit deployment!")


if __name__ == "__main__":
    main()
