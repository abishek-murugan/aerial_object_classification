import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.config import MODELS_DIR, EPOCHS
from src.data_loader import get_data_generators
from src.models import build_custom_cnn, build_transfer_model

def plot_history(history, model_name):
    """
    Plots training loss and accuracy metrics over epoch counts.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Loss')
    
    save_path = os.path.join(MODELS_DIR, f'{model_name}_history.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def train_model(model_type='custom'):
    """
    Executes model training with ModelCheckpoint and EarlyStopping callbacks.
    """
    print(f"\nStarting training for {model_type} model...")
    
    train_gen, valid_gen, _ = get_data_generators()
    
    if model_type == 'custom':
        model = build_custom_cnn()
    else:
        model = build_transfer_model()
        
    checkpoint_path = os.path.join(MODELS_DIR, f'{model_type}_model.keras')
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks
    )
    
    plot_history(history, model_type)
    print(f"{model_type} model training complete. Saved to {checkpoint_path}")

if __name__ == "__main__":
    train_model('custom')
    train_model('transfer')
