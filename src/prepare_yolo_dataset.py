import os
import shutil
from ultralytics import YOLO
import yaml

def prepare_yolo_dataset():
    """
    Transforms the classification image folder structure into YOLOv8 detection format.
    
    Generates relative label mappings (.txt) and configurations (.yaml) dynamically.
    Utilizes an existing trained YOLO model to auto-annotate if available,
    falling back to standardized central bounding boxes.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    class_data_dir = os.path.join(base_dir, 'dataset')
    yolo_data_dir = os.path.join(base_dir, 'object_detection_Dataset')
    
    if not os.path.exists(class_data_dir):
        print(f"Classification dataset not found at {class_data_dir}!")
        return

    print("Preparing YOLOv8 dataset...")
    
    splits = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }
    
    for split_name in splits.values():
        os.makedirs(os.path.join(yolo_data_dir, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(yolo_data_dir, 'labels', split_name), exist_ok=True)
        
    trained_yolo_path = os.path.join(base_dir, 'models', 'yolov8_results', 'weights', 'best.pt')
    model = None
    if os.path.exists(trained_yolo_path):
        try:
            model = YOLO(trained_yolo_path)
            print("Loaded trained YOLOv8 model for auto-labeling.")
        except Exception as e:
            print(f"Error loading trained YOLOv8 model: {e}")
            
    classes = {'bird': 0, 'drone': 1}
    
    for class_split, yolo_split in splits.items():
        print(f"Processing split: {class_split} -> {yolo_split}")
        
        for class_name, class_id in classes.items():
            class_folder = os.path.join(class_data_dir, class_split, class_name)
            if not os.path.exists(class_folder):
                continue
                
            files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in files:
                src_image_path = os.path.join(class_folder, file)
                dst_image_path = os.path.join(yolo_data_dir, 'images', yolo_split, file)
                
                shutil.copy2(src_image_path, dst_image_path)
                
                label_filename = os.path.splitext(file)[0] + '.txt'
                label_path = os.path.join(yolo_data_dir, 'labels', yolo_split, label_filename)
                
                bboxes = []
                
                if model is not None:
                    try:
                        results = model(src_image_path, verbose=False)
                        for box in results[0].boxes:
                            cls = int(box.cls[0].item())
                            xywh = box.xywhn[0].tolist()
                            bboxes.append(f"{cls} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}")
                    except Exception as e:
                        print(f"Error auto-labeling {file}: {e}")
                        
                if not bboxes:
                    bboxes.append(f"{class_id} 0.5 0.5 0.7 0.7")
                    
                with open(label_path, 'w') as lf:
                    lf.write('\n'.join(bboxes) + '\n')
                    
    data_yaml = {
        'path': yolo_data_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'bird',
            1: 'drone'
        }
    }
    
    yaml_path = os.path.join(yolo_data_dir, 'data_absolute.yaml')
    with open(yaml_path, 'w') as yf:
        yaml.dump(data_yaml, yf)
        
    print(f"YOLOv8 dataset preparation complete! Created config at {yaml_path}")

if __name__ == "__main__":
    prepare_yolo_dataset()
