#!/usr/bin/env python3
"""Test threshold tuning functionality"""

from pathlib import Path
import sys

# Add the web_ui directory to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "web_ui"))

from threshold_tuning import evaluate_thresholds

def main():
    # Test parameters
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    data_path = Path("data/beir_data")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"Testing threshold tuning with model: {model_name}")
    print(f"Data path: {data_path}")
    print(f"Thresholds: {thresholds}")
    
    def progress_callback(msg):
        print(f"Progress: {msg}")
    
    try:
        results = evaluate_thresholds(
            model_name=model_name,
            data_path=data_path,
            thresholds=thresholds,
            batch_size=32,
            use_filtered_corpus=True,
            progress_callback=progress_callback,
            max_queries=10  # Test with only 10 queries for speed
        )
        
        print("\nResults:")
        print(f"Best threshold: {results['best_threshold']:.3f}")
        print(f"Best F1: {results['best_f1']:.3f}")
        print(f"Best precision: {results['best_precision']:.3f}")
        print(f"Best recall: {results['best_recall']:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()