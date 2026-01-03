# genai.py - Updated with truly free models
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
# 3. Load a truly free, small LLM
# -------------------------
print("⏳ Loading small local LLM...")

# List of truly free, small models (no authentication required)
FREE_MODELS = [
    "microsoft/phi-1_5",      # 1.3B parameters (~2.6GB)
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B parameters
    "HuggingFaceTB/SmolLM2-135M",  # Very small 135M parameters
    "google/recurrentgemma-2b",  # Alternative to Gemma (may work)
]

llm_pipeline = None
loaded_model_name = None

for model_id in FREE_MODELS:
    try:
        print(f"  Trying {model_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Create text generation pipeline
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,  # Use CPU
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        loaded_model_name = model_id
        print(f"✅ Successfully loaded: {model_id}")
        break
        
    except Exception as e:
        print(f"  ⚠️  Failed to load {model_id}: {str(e)[:100]}...")
        continue

if llm_pipeline is None:
    print("⚠️  Could not load any local LLM, using template reports")

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

print("\n" + "="*60)
print("🔍 FAULT DETECTION RESULTS")
print("="*60)
print(f"📊 Detected fault: {fault_type}")
print(f"🎯 Confidence: {fault_confidence:.2%}")
print(f"💾 Using LLM: {loaded_model_name if loaded_model_name else 'Template'}")

# Display confidence for all classes
print("\n📈 All Class Probabilities:")
for cls, prob in zip(classes, pred):
    bar_length = int(prob * 30)
    bar = "█" * bar_length + "░" * (30 - bar_length)
    print(f"  {cls:15s} {prob:.4f} {bar}")

# -------------------------
# 7. OPTION A: Use OpenAI API (FREE Tier - $0.00)
# -------------------------
def generate_with_openai_api(prompt):
    """Use OpenAI API with free credits"""
    try:
        # You need to install: pip install openai
        from openai import OpenAI
        
        # Initialize with your API key
        # Get free credits: https://platform.openai.com/
        client = OpenAI(api_key="your-api-key-here")  # Replace with your key
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a solar panel maintenance expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except ImportError:
        print("⚠️  OpenAI package not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"⚠️  OpenAI API error: {e}")
        return None

# -------------------------
# 8. OPTION B: Use Google Gemini API (FREE)
# -------------------------
def generate_with_gemini_api(prompt):
    """Use Google Gemini API (free tier)"""
    try:
        # Install: pip install google-generativeai
        import google.generativeai as genai
        
        # Configure with your API key
        # Get key: https://makersuite.google.com/app/apikey
        genai.configure(api_key="your-api-key-here")  # Replace with your key
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
        
    except ImportError:
        print("⚠️  Google Generative AI package not installed. Install with: pip install google-generativeai")
        return None
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return None

# -------------------------
# 9. OPTION C: Use Together AI (FREE Credits)
# -------------------------
def generate_with_together_api(prompt):
    """Use Together AI (free $10 credits)"""
    try:
        import requests
        import json
        
        # Get free credits: https://www.together.ai/
        api_key = "your-api-key-here"  # Replace with your key
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "messages": [
                    {"role": "system", "content": "You are a solar panel maintenance expert."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"⚠️  Together API error: {response.text}")
            return None
            
    except Exception as e:
        print(f"⚠️  Together API error: {e}")
        return None

# -------------------------
# 10. Generate maintenance report
# -------------------------
def create_maintenance_prompt(fault_type, confidence, sensor_data):
    """Create a structured prompt"""
    return f"""Create a solar panel maintenance report for a {fault_type} fault.

DETECTED ISSUE: {fault_type}
CONFIDENCE: {confidence:.1%}

SENSOR DATA:
- Voltage: {sensor_data['Voltage']}V
- Current: {sensor_data['Current']}A  
- Temperature: {sensor_data['Temperature']}°C
- Humidity: {sensor_data['Humidity']}%
- Pressure: {sensor_data['Pressure']}hPa

Please provide:
1. Brief explanation of {fault_type} issue
2. How it affects performance
3. Severity level (Low/Medium/High)
4. Recommended actions
5. Timeline for addressing

Keep it concise and practical for field technicians."""

def generate_with_local_llm(prompt):
    """Generate with local LLM if available"""
    if llm_pipeline is None:
        return None
    
    try:
        response = llm_pipeline(
            prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return response[0]['generated_text']
    except Exception as e:
        print(f"⚠️  Local LLM error: {e}")
        return None

# -------------------------
# 11. Generate and display report
# -------------------------
print("\n" + "="*60)
print("🤖 GENERATING MAINTENANCE REPORT")
print("="*60)

prompt = create_maintenance_prompt(fault_type, fault_confidence, sensor_data)

# Try different generation methods in order
report = None
method = ""

# Method 1: Try local LLM first
if llm_pipeline:
    print("🔄 Method 1: Trying local LLM...")
    report = generate_with_local_llm(prompt)
    if report:
        method = "Local LLM"
    else:
        print("  ⚠️  Local LLM failed")

# Method 2: Try API options (uncomment and add your API keys)
if not report:
    print("🔄 Method 2: Trying API options...")
    # Uncomment ONE of these and add your API key:
    
    # Option A: OpenAI (requires API key)
    # report = generate_with_openai_api(prompt)
    # if report: method = "OpenAI API"
    
    # Option B: Gemini (requires API key)  
    # report = generate_with_gemini_api(prompt)
    # if report: method = "Gemini API"
    
    # Option C: Together AI (requires API key)
    # report = generate_with_together_api(prompt)
    # if report: method = "Together AI"

# Method 3: Fallback template
if not report:
    print("🔄 Method 3: Using template report")
    
    # Smart template based on fault type
    templates = {
        'Dust': f"""
SOLAR PANEL MAINTENANCE REPORT
Generated: {fault_confidence:.1%} confidence

ISSUE: Dust Accumulation
- Dust layer reduces light transmission
- Estimated efficiency loss: 15-25%
- Common in dry, arid environments

SENSOR ANALYSIS:
• Voltage ({sensor_data['Voltage']}V): Below optimal range
• Current ({sensor_data['Current']}A): Reduced output
• Temperature ({sensor_data['Temperature']}°C): Normal
• Humidity ({sensor_data['Humidity']}%): Moderate

SEVERITY: {'HIGH' if fault_confidence > 0.8 else 'MEDIUM'}

ACTIONS REQUIRED:
1. IMMEDIATE: Schedule cleaning within 1 week
2. Clean with soft brush and deionized water
3. Test output post-cleaning
4. Document in maintenance log

PREVENTIVE MEASURES:
• Establish monthly cleaning schedule
• Consider automated cleaning system
• Monitor dust accumulation rate
""",
        'Crack': """
SOLAR PANEL MAINTENANCE REPORT - URGENT

ISSUE: Panel Cracking
- Physical damage to solar cells
- Risk of water ingress and electrical issues
- Requires immediate attention

SEVERITY: HIGH

ACTIONS REQUIRED:
1. IMMEDIATE: Isolate panel from system
2. Schedule professional inspection
3. Check warranty coverage
4. Plan for panel replacement

SAFETY PRECAUTIONS:
• Do not touch damaged panel
• Disconnect from inverter
• Mark as out of service
""",
        'Burn': """
SOLAR PANEL MAINTENANCE REPORT - URGENT

ISSUE: Burn Marks/Hot Spots
- Indicates electrical issues or cell damage
- Can lead to fire risk if untreated
- Requires electrical inspection

SEVERITY: HIGH

ACTIONS REQUIRED:
1. IMMEDIATE: Power down affected panel
2. Schedule electrical inspection
3. Check junction box and connectors
4. Test bypass diodes

SAFETY FIRST:
• Do not operate damaged panel
• Check for smoke or heat
• Ensure proper grounding
""",
        'Delamination': """
SOLAR PANEL MAINTENANCE REPORT

ISSUE: Delamination
- Separation of layers in solar panel
- Allows moisture ingress
- Reduces lifespan and efficiency

SEVERITY: MEDIUM

ACTIONS REQUIRED:
1. Schedule professional assessment
2. Check manufacturing warranty
3. Monitor for spreading
4. Consider replacement if expanding

PREVENTIVE:
• Regular visual inspections
• Document any changes
• Monitor performance degradation
""",
        'Normal': """
SOLAR PANEL MAINTENANCE REPORT

STATUS: Normal Operation
- All systems functioning properly
- No faults detected
- Regular maintenance recommended

SEVERITY: NONE

ROUTINE MAINTENANCE:
1. Continue regular monitoring
2. Clean panels quarterly
3. Check connections annually
4. Update maintenance logs

RECOMMENDED:
• Document baseline performance
• Schedule next inspection
• Monitor for any changes
"""
    }
    
    report = templates.get(fault_type, templates['Normal'])
    method = "Smart Template"

print(f"\n📄 MAINTENANCE REPORT ({method}):")
print("="*40)
print(report)

# -------------------------
# 12. Generate quick summary
# -------------------------
print("\n" + "="*60)
print("📋 EXECUTIVE SUMMARY")
print("="*60)

summary = f"""
🔴 DETECTED FAULT: {fault_type}
🎯 CONFIDENCE: {fault_confidence:.1%}
📊 METHOD: {method}

🚨 URGENCY: {'HIGH' if fault_type in ['Crack', 'Burn'] else 'MEDIUM' if fault_confidence > 0.7 else 'LOW'}
⏰ ACTION TIMELINE: {'IMMEDIATE (1-3 days)' if fault_type in ['Crack', 'Burn'] else 'SOON (1-2 weeks)' if fault_type == 'Dust' else 'ROUTINE (1 month)'}
💰 ESTIMATED IMPACT: {'15-25% efficiency loss' if fault_type == 'Dust' else '30-50% efficiency loss' if fault_type in ['Crack', 'Burn'] else '5-15% efficiency loss'}

✅ NEXT STEPS:
1. {'Schedule emergency repair' if fault_type in ['Crack', 'Burn'] else 'Schedule cleaning'}
2. Document findings
3. Update maintenance schedule
4. Monitor post-repair performance
"""

print(summary)

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE")
print("="*60)

# -------------------------
# 13. Save report to file
# -------------------------
def save_report_to_file(report_text, filename="solar_maintenance_report.txt"):
    """Save the report to a text file"""
    try:
        timestamp = tf.timestamp().numpy()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"SOLAR PANEL MAINTENANCE REPORT\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Fault: {fault_type} (Confidence: {fault_confidence:.1%})\n")
            f.write(f"Method: {method}\n")
            f.write("="*50 + "\n\n")
            f.write(report_text)
            f.write("\n\n" + "="*50 + "\n")
            f.write(f"Sensor Data: {sensor_data}\n")
        
        print(f"\n💾 Report saved to: {filename}")
        return True
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")
        return False

# Save the report
save_report_to_file(report)