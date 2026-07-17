import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import MODELS_DIR, RESULTS_DIR, CLASSES
from src.data_loader import get_data_generators

def evaluate_model(model_type='custom'):
    """
    Evaluates saved classifier models on the hold-out test set and saves the confusion matrix.
    """
    print(f"Evaluating {model_type} model...")
    
    _, _, test_gen = get_data_generators()
    
    model_path = os.path.join(MODELS_DIR, f'{model_type}_model.keras')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
        
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(test_gen)
    
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    print(f"\n--- {model_type.upper()} MODEL REPORT ---")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_type} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(RESULTS_DIR, f'{model_type}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    evaluate_model('custom')
    evaluate_model('transfer')
