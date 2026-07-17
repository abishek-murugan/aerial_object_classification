import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import cv2
if not hasattr(cv2, 'imshow'):
    cv2.imshow = lambda *args, **kwargs: None
from ultralytics import YOLO
import os
from src.config import MODELS_DIR, RESULTS_DIR, YOLO_DATA_DIR

def train_yolo():
    """
    Fine-tunes a pre-trained YOLOv8 Nano model on the custom aerial targets dataset.
    """
    print("Starting YOLOv8 training...")
    
    model = YOLO('yolov8n.pt')
    data_yaml = os.path.join(YOLO_DATA_DIR, 'data_absolute.yaml')
    
    results = model.train(
        data=data_yaml,
        epochs=2,
        imgsz=320,
        project=RESULTS_DIR,
        name='yolov8_results',
        exist_ok=True,
        device='cpu',
        workers=2
    )
    
    print("YOLOv8 training complete.")
    
    metrics = model.val()
    print("Validation Metrics:", metrics)
    
    try:
        success = model.export(format='onnx')
        print("Model exported to ONNX:", success)
    except Exception as e:
        print(f"ONNX export skipped/failed: {e} (PyTorch weights are successfully saved)")

if __name__ == "__main__":
    train_yolo()
