# PoC: Solar Panel Fault Detection + Explanation using GPT-Neo 1.3B
# For a single image with proper GPT-Neo generation

import tensorflow as tf
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# -------------------------
# 1. Load Keras model (instead of TFLite for compatibility with trained model)
# -------------------------
keras_model_path = r"solar_fault_model.h5"

if not os.path.exists(keras_model_path):
    raise FileNotFoundError(f"Keras model not found at {keras_model_path}")

cnn_model = tf.keras.models.load_model(keras_model_path)

print("Keras model loaded successfully.")

# -------------------------
# 2. Load GPT-Neo 1.3B
# -------------------------
model_name = "EleutherAI/gpt-neo-1.3B"
print("Loading GPT-Neo 1.3B (this may take a while)...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
gpt_model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=gpt_model, tokenizer=tokenizer)

print("GPT-Neo loaded successfully.")

# -------------------------
# 3. Define sensor data
# -------------------------
sensor_data = {
    "Voltage": 17.8,       # in Volts
    "Current": 4.3,        # in Amps
    "Temperature": 42,     # in Celsius
    "Humidity": 65,        # in %
    "Pressure": 1012       # in hPa
}

# -------------------------
# 4. Function to detect fault from image
# -------------------------
def detect_fault(image_path):
    if not os.path.exists(image_path):
        print("Error: Image path does not exist:", image_path)
        return None

    print("Loading image:", image_path)
    img = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(np.array(img, dtype=np.float32)/255.0, axis=0)

    prediction = cnn_model.predict(input_data)
    print("Raw prediction output:", prediction)

    classes = ['Normal', 'Crack', 'Burn', 'Dust', 'Delamination']
    fault_type = classes[np.argmax(prediction)]
    return fault_type

# -------------------------
# 5. Function to generate explanation via GPT-Neo
# -------------------------
def generate_report(fault_type, sensor_data):
    prompt = f"""
You are an AI solar maintenance assistant.

Detected fault (from vision model): {fault_type}
Sensor readings:
- Voltage: {sensor_data['Voltage']} V
- Current: {sensor_data['Current']} A
- Temperature: {sensor_data['Temperature']} °C
- Humidity: {sensor_data['Humidity']} %
- Pressure: {sensor_data['Pressure']} hPa

Tasks:
1. Explain the detected fault in simple technical terms.
2. Correlate image-based fault with sensor readings.
3. State severity level (Low / Medium / High).
4. Give preventive and corrective maintenance actions.

Answer clearly using bullet points.
"""

    print("Generating GPT-Neo report...")
    output = generator(
        prompt,
        max_length=512,          # increase length for detailed report
        do_sample=True,          # allow creativity
        temperature=0.7,         # controls randomness
        pad_token_id=tokenizer.eos_token_id
    )[0]['generated_text']

    return output

# -------------------------
# 6. Single image processing
# -------------------------
image_path = r"D:\sudo-solo\dataset\dusty\dusty_1.jpg"  # Replace with your image path

fault = detect_fault(image_path)
if fault:
    print("\nDetected Fault:", fault)
    report = generate_report(fault, sensor_data)
    print("\nGPT-Neo Report:\n", report)
