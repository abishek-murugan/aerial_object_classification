import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, VALID_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def get_data_generators():
    """
    Initializes and returns ImageDataGenerators for the classification pipeline.
    
    Applies real-time augmentation on the training set for regularization
    and rescales the validation and test sets.
    
    Returns:
        tuple: (train_generator, valid_generator, test_generator)
    """
    # Training augmentations to improve generalization on unseen aerial targets
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation/Test loaders only require scaling
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # Stream splits from local directories
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )

    valid_generator = valid_test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = valid_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator
