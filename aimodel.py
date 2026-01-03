# aimodel.py
# PoC: Solar Panel Fault Detection + Explanation using CNN + Flan-T5

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings

# Suppress TensorFlow warnings if desired
warnings.filterwarnings('ignore', category=UserWarning)

# -------------------------
# 1. Paths
# -------------------------
CNN_MODEL_PATH = r"D:\sudo-solo\solar_fault_model.h5"   # Your trained model
IMAGE_PATH = r"D:\sudo-solo\dataset\dusty\dusty_1.jpg"  # Replace with your test image

# -------------------------
# 2. Load CNN model
# -------------------------
if not os.path.exists(CNN_MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {CNN_MODEL_PATH}")

cnn_model = load_model(CNN_MODEL_PATH)
print("✅ CNN model loaded")
print("📐 CNN input shape:", cnn_model.input_shape)

# -------------------------
# 3. Load Flan-T5 model
# -------------------------
print("⏳ Loading Flan-T5 model...")
model_name = "google/flan-t5-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=flan_model, tokenizer=tokenizer)
    print("✅ Flan-T5 loaded")
except Exception as e:
    print(f"⚠️  Failed to load Flan-T5: {e}")
    print("⚠️  Using fallback report generation")
    generator = None

# -------------------------
# 4. Sensor data
# -------------------------
sensor_data = {
    "Voltage": 17.8,       # Volts
    "Current": 4.3,        # Amps
    "Temperature": 42,     # Celsius
    "Humidity": 65,        # %
    "Pressure": 1012       # hPa
}

# -------------------------
# 5. Image preprocessing
# -------------------------
def preprocess_image(img_path, target_size):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Image not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

# -------------------------
# 6. Fault detection
# -------------------------
classes = ['Normal', 'Crack', 'Burn', 'Dust', 'Delamination']

# Detect model's expected input size
input_shape = cnn_model.input_shape[1:3]  # (height, width)
image = preprocess_image(IMAGE_PATH, input_shape)

pred = cnn_model.predict(image, verbose=0)[0]
fault_idx = np.argmax(pred)
fault_confidence = pred[fault_idx]
fault_type = classes[fault_idx]

print("\n" + "="*50)
print("🔍 FAULT DETECTION RESULTS")
print("="*50)
print(f"Prediction probabilities: {pred}")
print(f"Detected fault: {fault_type}")
print(f"Confidence: {fault_confidence:.2%}")
print(f"Confidence (raw): {fault_confidence:.4f}")

# Display all probabilities
print("\n📊 All Class Probabilities:")
for i, (cls, prob) in enumerate(zip(classes, pred)):
    bar = "█" * int(prob * 40)
    print(f"{cls:15s} {prob:.4f} {bar}")

# -------------------------
# 7. Generate maintenance report
# -------------------------
def generate_maintenance_report(fault_type, confidence, sensor_data):
    """Generate maintenance report using Flan-T5 or fallback"""
    
    prompt = f"""
Generate a comprehensive maintenance report for a solar panel with the following structure:

1. **Fault Explanation**
2. **Correlation with Sensor Readings**
3. **Severity Level** (Low/Medium/High)
4. **Preventive Actions**
5. **Corrective Actions**

Detected fault: {fault_type} (Confidence: {confidence:.2f})

Sensor readings:
- Voltage: {sensor_data['Voltage']} V
- Current: {sensor_data['Current']} A
- Temperature: {sensor_data['Temperature']} °C
- Humidity: {sensor_data['Humidity']} %
- Pressure: {sensor_data['Pressure']} hPa

Write in clear bullet points.
"""
    
    if generator:
        try:
            report = generator(
                prompt,
                max_new_tokens=350,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )[0]['generated_text']
            return report
        except Exception as e:
            print(f"⚠️  Error generating report: {e}")
            return generate_fallback_report(fault_type, confidence, sensor_data)
    else:
        return generate_fallback_report(fault_type, confidence, sensor_data)

def generate_fallback_report(fault_type, confidence, sensor_data):
    """Fallback report if Flan-T5 fails"""
    severity = "Medium" if confidence > 0.7 else "Low"
    
    report = f"""
1. **Fault Explanation**
   - Detected {fault_type.lower()} accumulation on solar panel surface
   - Reduces light transmission to photovoltaic cells
   - Common in arid/dusty environments

2. **Correlation with Sensor Readings**
   - Voltage: {sensor_data['Voltage']}V (normal range: 17-22V)
   - Current: {sensor_data['Current']}A (slightly reduced)
   - Temperature: {sensor_data['Temperature']}°C (within normal range)
   - Humidity: {sensor_data['Humidity']}% (moderate, may contribute to dust adhesion)

3. **Severity Level**: {severity}

4. **Preventive Actions**
   - Regular cleaning schedule (bi-weekly/monthly)
   - Install anti-soiling coatings
   - Consider automated cleaning systems
   - Monitor performance degradation rate

5. **Corrective Actions**
   - Perform manual cleaning with soft brush and deionized water
   - Inspect for any physical damage post-cleaning
   - Verify performance improvement post-maintenance
   - Document cleaning in maintenance log
"""
    return report

# -------------------------
# 8. Generate and display report
# -------------------------
print("\n" + "="*50)
print("📄 AI MAINTENANCE REPORT")
print("="*50)

report = generate_maintenance_report(fault_type, fault_confidence, sensor_data)
print(report)

# -------------------------
# 9. Additional recommendations
# -------------------------
print("\n" + "="*50)
print("💡 ADDITIONAL RECOMMENDATIONS")
print("="*50)

recommendations = {
    'Dust': [
        "Schedule cleaning within 1-2 weeks",
        "Check weather forecasts for optimal cleaning window",
        "Consider rain-assisted natural cleaning"
    ],
    'Crack': [
        "Immediate inspection required",
        "Consider panel replacement",
        "Check warranty coverage"
    ],
    'Burn': [
        "Investigate electrical issues",
        "Check junction box and connectors",
        "Verify grounding system"
    ],
    'Delamination': [
        "Check manufacturing warranty",
        "Evaluate moisture ingress",
        "Consider professional assessment"
    ],
    'Normal': [
        "Continue regular monitoring",
        "Document baseline performance",
        "Schedule next routine inspection"
    ]
}

for rec in recommendations.get(fault_type, []):
    print(f"• {rec}")

print("\n" + "="*50)
print("✅ Analysis Complete")
print("="*50)