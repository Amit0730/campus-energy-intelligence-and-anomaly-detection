"""
Campus Energy Pipeline Training
"""

import sys
import os

# root code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import CampusEnergyPipeline

def main():
    print("Training Campus Energy Intelligence Pipeline...")
    
    pipeline = CampusEnergyPipeline(data_dir="data", model_dir="saved_models")
    metrics = pipeline.run_full_pipeline(num_days=730, seed=42)
    
    print("\n--- Training Results ---")
    print(f"Best Model: {metrics['best_model']}")
    print(f"Total Features: {metrics['feature_count']}")
    print(f"Total Records: {metrics['num_records_total']}")
    print(f"Test Records: {metrics['num_test_records']}")
    print(f"Anomaly Precision: {metrics['anomaly_metrics']['Precision']:.4f}")
    print(f"Anomaly Recall: {metrics['anomaly_metrics']['Recall']:.4f}")
    print(f"Anomaly F1: {metrics['anomaly_metrics']['F1_Score']:.4f}")
    print("\nPipeline run complete.")


if __name__ == "__main__":
    main()
