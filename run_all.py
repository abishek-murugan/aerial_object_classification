import os
import subprocess
import sys

def run_module(module_name):
    """
    Executes a Python module as a separate process.
    """
    print("=" * 50)
    print(f"Executing: {module_name}...")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, '-m', module_name], check=True)
        print(f"Successfully executed {module_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Pipeline error in module {module_name}: {e}")
        sys.exit(1)

def main():
    """
    Orchestrates the training, evaluation, data preparation, and fine-tuning pipeline.
    """
    print("AeroEye AI Pipeline Initialization...\n")
    
    # 1. Classification Model Training
    run_module('src.train_classification')
    
    # 2. Performance Evaluation
    run_module('src.evaluate_classification')
    
    # 3. Object Detection Dataset Formatting
    run_module('src.prepare_yolo_dataset')
    
    # 4. YOLOv8 Model Fine-Tuning
    run_module('src.train_yolo')
    
    print("=" * 50)
    print("End-to-End Pipeline Execution Successful!")
    print("Launch dashboard: streamlit run app.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
