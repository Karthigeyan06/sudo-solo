import tensorflow as tf
import numpy as np
from PIL import Image

# Load the Keras model instead of TFLite
model = tf.keras.models.load_model('solar_fault_model.h5')

img = Image.open(r"D:\sudo-solo\dataset\dusty\dusty_19.jpg").resize((224, 224))
input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

prediction = model.predict(input_data)
classes = ['Burn', 'Crack', 'Delamination', 'Dust', 'Normal']
result = classes[np.argmax(prediction)]
print("Detected:", result)
