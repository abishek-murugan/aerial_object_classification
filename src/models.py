import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from src.config import IMG_SIZE

def build_custom_cnn(input_shape=IMG_SIZE + (3,)):
    """
    Constructs a lightweight custom CNN architecture optimized for CPU training.
    
    Includes global average pooling to drastically reduce classification head parameter count,
    minimizing overfitting risk and latency.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Dense feature reduction
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_transfer_model(input_shape=IMG_SIZE + (3,)):
    """
    Constructs a MobileNetV2 transfer learning model.
    
    Initializes pre-trained weights from ImageNet with frozen base features,
    feeding into a custom classification head.
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
