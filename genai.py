# aimodel.py - Updated with local Gemma model
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings

warnings.filterwarnings('ignore')

# -------------------------
# 1. Paths
# -------------------------
CNN_MODEL_PATH = r"D:\sudo-solo\solar_fault_model.h5"
IMAGE_PATH = r"D:\sudo-solo\dataset\dusty\dusty_1.jpg"

# -------------------------
# 2. Load CNN model
# -------------------------
if not os.path.exists(CNN_MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {CNN_MODEL_PATH}")

cnn_model = load_model(CNN_MODEL_PATH)
print("✅ CNN model loaded")
print("📐 CNN input shape:", cnn_model.input_shape)

# -------------------------
# 3. Load Local LLM (Gemma-2B)
# -------------------------
print("⏳ Loading local LLM (Gemma-2B)...")

try:
    # Use CPU to avoid GPU memory issues
    model_id = "google/gemma-2b"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU stability
        device_map="cpu",  # Force CPU usage
        low_cpu_mem_usage=True
    )
    
    # Create text generation pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # Use CPU
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    print("✅ Local LLM loaded (Gemma-2B on CPU)")
    
except Exception as e:
    print(f"⚠️  Could not load Gemma: {e}")
    print("🔄 Falling back to Phi-2...")
    try:
        model_id = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True
        )
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_new_tokens=300
        )
        print("✅ Local LLM loaded (Phi-2 on CPU)")
    except:
        print("⚠️  Could not load any local LLM")
        llm_pipeline = None

# -------------------------
# 4. Sensor data
# -------------------------
sensor_data = {
    "Voltage": 17.8,
    "Current": 4.3,
    "Temperature": 42,
    "Humidity": 65,
    "Pressure": 1012
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
input_shape = cnn_model.input_shape[1:3]
image = preprocess_image(IMAGE_PATH, input_shape)

pred = cnn_model.predict(image, verbose=0)[0]
fault_idx = np.argmax(pred)
fault_confidence = pred[fault_idx]
fault_type = classes[fault_idx]

print("\n" + "="*50)
print("🔍 FAULT DETECTION RESULTS")
print("="*50)
print(f"Detected fault: {fault_type}")
print(f"Confidence: {fault_confidence:.2%}")

# -------------------------
# 7. Generate maintenance report
# -------------------------
def generate_with_local_llm(prompt):
    """Generate text using local LLM"""
    if llm_pipeline is None:
        return None
    
    try:
        response = llm_pipeline(
            prompt,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        return response[0]['generated_text']
    except Exception as e:
        print(f"⚠️  Generation error: {e}")
        return None

def create_maintenance_prompt(fault_type, confidence, sensor_data):
    """Create a structured prompt for the LLM"""
    return f"""You are a solar panel maintenance expert. Create a detailed maintenance report.

DETECTED FAULT: {fault_type}
CONFIDENCE LEVEL: {confidence:.1%}

SENSOR READINGS:
- Voltage: {sensor_data['Voltage']} V
- Current: {sensor_data['Current']} A
- Temperature: {sensor_data['Temperature']} °C
- Humidity: {sensor_data['Humidity']} %
- Pressure: {sensor_data['Pressure']} hPa

REPORT STRUCTURE:
1. FAULT EXPLANATION: Explain what {fault_type} means for solar panels
2. SENSOR CORRELATION: How the sensor readings relate to this fault
3. SEVERITY ASSESSMENT: Low/Medium/High with justification
4. PREVENTIVE ACTIONS: 3-5 specific preventive measures
5. CORRECTIVE ACTIONS: 3-5 specific corrective steps
6. TIMELINE: Urgency and suggested timeline

Write in clear, professional language suitable for maintenance technicians."""

def generate_fallback_report(fault_type, confidence, sensor_data):
    """Fallback report if LLM fails"""
    severity = "Medium" if confidence > 0.7 else "Low"
    
    report = f"""
MAINTENANCE REPORT - SOLAR PANEL FAULT DETECTION

1. FAULT EXPLANATION
   • Detected: {fault_type}
   • Confidence: {confidence:.1%}
   • Impact: Reduces panel efficiency and power output

2. SENSOR CORRELATION
   • Voltage ({sensor_data['Voltage']}V): Slightly below optimal range
   • Current ({sensor_data['Current']}A): Indicative of reduced output
   • Temperature ({sensor_data['Temperature']}°C): Normal operating range
   • Humidity ({sensor_data['Humidity']}%): May accelerate degradation

3. SEVERITY ASSESSMENT: {severity}
   • {fault_type} accumulation reduces efficiency by 15-25%

4. PREVENTIVE ACTIONS
   • Establish regular cleaning schedule
   • Install dust monitoring sensors
   • Apply anti-soiling coatings
   • Trim surrounding vegetation

5. CORRECTIVE ACTIONS
   • Perform thorough cleaning with soft brush
   • Use deionized water for final rinse
   • Test panel output post-cleaning
   • Document maintenance in system log

6. TIMELINE
   • Recommended action within: 1-2 weeks
   • Follow-up inspection: 1 month
"""
    return report

# -------------------------
# 8. Generate and display report
# -------------------------
print("\n" + "="*50)
print("🤖 GENERATING AI MAINTENANCE REPORT")
print("="*50)

prompt = create_maintenance_prompt(fault_type, fault_confidence, sensor_data)

if llm_pipeline:
    print("⏳ Generating report with local LLM...")
    ai_report = generate_with_local_llm(prompt)
    
    if ai_report:
        print("\n📄 AI-GENERATED MAINTENANCE REPORT:")
        print("-" * 40)
        print(ai_report)
    else:
        print("⚠️  Using fallback report (LLM failed)")
        print("\n📄 MAINTENANCE REPORT:")
        print("-" * 40)
        print(generate_fallback_report(fault_type, fault_confidence, sensor_data))
else:
    print("⚠️  No LLM available, using template report")
    print("\n📄 MAINTENANCE REPORT:")
    print("-" * 40)
    print(generate_fallback_report(fault_type, fault_confidence, sensor_data))

# -------------------------
# 9. Install dependencies (requirements.txt)
# -------------------------
"""
Create a requirements.txt file:

transformers>=4.35.0
torch>=2.0.0
tensorflow>=2.10.0
pillow>=9.0.0
numpy>=1.24.0
accelerate>=0.24.0

Install with: pip install -r requirements.txt
"""

print("\n" + "="*50)
print("✅ ANALYSIS COMPLETE")
print("="*50)