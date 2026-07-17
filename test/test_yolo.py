import tensorflow as tf
import keras_cv
import numpy as np

print("Loading model...")
model = keras_cv.models.YOLOV8Detector.from_preset("yolo_v8_m_pascalvoc", bounding_box_format="xyxy")

print("Creating dummy image...")
img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
img_tensor = tf.expand_dims(img_tensor, axis=0)

print("Predicting...")
results = model.predict(img_tensor, verbose=0)
print(results.keys())
print(results['boxes'].shape)
print("Success!")
