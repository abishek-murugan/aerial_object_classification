import os

def download_models():
    """
    Checks for the existence of required models in the models output folder.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    required_models = {
        'custom_model.keras': 'Custom CNN Classifier',
        'transfer_model.keras': 'Transfer Learning Classifier',
        'yolov8_results/weights/best.pt': 'YOLOv8 Object Detector'
    }

    print("=" * 50)
    print("Verifying Model Status...")
    print("=" * 50)

    missing_any = False
    for filename, name in required_models.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            print(f"✔️ {name:30} - FOUND ({path})")
        else:
            print(f"❌ {name:30} - MISSING")
            missing_any = True

    if missing_any:
        print("\nNote: Some models are missing. Run the training script:")
        print("      python run_all.py")
    else:
        print("\nAll models are ready for the Streamlit dashboard!")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    download_models()
