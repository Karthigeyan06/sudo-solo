"""
SUDO-SOLO Automation Pipeline Configuration
Central configuration for all automation components
"""

import os
from pathlib import Path

# ======================= PROJECT PATHS =======================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images_captured')

# ======================= NETWORK CONFIGURATION =======================
# Control center IP and ports
CONTROL_CENTER_IP = "192.168.1.100"  # Update with your machine IP
CONTROL_CENTER_PORT = 5000

# Device-specific ports
DEVICE_CONTROL_PORT = 5001  # For device.cpp control
CAMERA_SERVER_PORT = 80     # ESP32-CAM web server
DETECTION_SERVER_PORT = 5000  # detect.py Flask server
GENAI_SERVER_PORT = 5002    # GenAI report generation

# ======================= SYSTEM MODE =======================
AUTONOMOUS_MODE_ENABLED = True  # Set False to disable automation

# ======================= SCHEDULING CONFIGURATION =======================
CLEANING_INTERVAL_DAYS = 7  # Weekly cleaning
FAULT_CHECK_INTERVAL_HOURS = 24  # Daily fault detection
MAINTENANCE_CHECK_HOUR = 6  # 6 AM daily

# ======================= SENSOR CONFIGURATION =======================
SENSOR_POLLING_INTERVAL = 300  # Poll every 5 minutes (seconds)

# Default sensor data (will be overridden by real sensors)
DEFAULT_SENSOR_DATA = {
    'Voltage': 17.8,
    'Current': 4.3,
    'Temperature': 42,
    'Humidity': 65,
    'Pressure': 1012
}

# Sensor thresholds for alerts
SENSOR_THRESHOLDS = {
    'voltage_min': 15.0,
    'voltage_max': 24.0,
    'current_min': 0.0,
    'current_max': 10.0,
    'temperature_max': 60.0,
    'humidity_max': 80.0,
}

# ======================= FAULT DETECTION CONFIGURATION =======================
FAULT_CONFIDENCE_THRESHOLD = 0.6  # 60% confidence minimum
CRITICAL_FAULTS = ['Burn', 'Delamination']  # Requires immediate action
HIGH_PRIORITY_FAULTS = ['Crack']
LOW_PRIORITY_FAULTS = ['Dust', 'Normal']

# Fault severity mapping
FAULT_SEVERITY = {
    'Burn': 'CRITICAL',
    'Delamination': 'CRITICAL',
    'Crack': 'HIGH',
    'Dust': 'MEDIUM',
    'Normal': 'LOW'
}

# ======================= ROBOT MOVEMENT CONFIGURATION =======================
# Cleaning sequence parameters
CLEANING_DURATION_SECONDS = 120  # 2 minutes per cleaning cycle
CLEANING_SPEED_PWM = 150  # PWM speed for cleaning (0-255)
MOVEMENT_TIMEOUT_SECONDS = 300  # Max 5 minutes per movement

# ======================= LOGGING CONFIGURATION =======================
LOG_FILE_PATH = os.path.join(LOGS_DIR, 'automation_pipeline.log')
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# Keep logs for this many days
LOG_RETENTION_DAYS = 30

# ======================= REPORT CONFIGURATION =======================
MAINTENANCE_REPORT_PATH = REPORTS_DIR
REPORT_FORMATS = ['json', 'txt', 'html']  # Supported formats

# Report retention
REPORT_RETENTION_DAYS = 365  # Keep reports for a year

# ======================= AI REPORT GENERATION =======================
# LLM Configuration
LLM_MODEL = "google/gemma-2b"  # or "microsoft/phi-2" as fallback
LLM_MAX_TOKENS = 350
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_DEVICE = "cpu"  # Use CPU for stability

# Enable/disable specific report sections
REPORT_SECTIONS = {
    'fault_explanation': True,
    'severity_assessment': True,
    'impact_analysis': True,
    'preventive_actions': True,
    'corrective_steps': True,
    'timeline': True,
    'sensor_correlation': True
}

# ======================= NOTIFICATION CONFIGURATION =======================
# Email notifications for critical faults
ENABLE_EMAIL_NOTIFICATIONS = False
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_FROM = "robot@sudo-solo.local"
EMAIL_RECIPIENTS = ["admin@example.com"]

# ======================= CAMERA CONFIGURATION =======================
CAMERA_RESOLUTION = (640, 480)
CAMERA_QUALITY = 80  # JPEG quality (0-100)
CAMERA_SNAPSHOT_TIMEOUT = 10  # seconds

# Image retention
IMAGE_RETENTION_DAYS = 30  # Keep captured images for this many days

# ======================= DATABASE CONFIGURATION =======================
USE_DATABASE = False  # Set to True if using database for reports
DB_TYPE = "sqlite"  # sqlite, postgresql, mysql
DB_PATH = os.path.join(PROJECT_ROOT, 'maintenance.db')

# ======================= API CONFIGURATION =======================
API_TIMEOUT = 30  # seconds
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# ======================= DEBUG & DEVELOPMENT =======================
DEBUG_MODE = False  # Enables verbose logging and debug output
SIMULATION_MODE = False  # Simulate device responses without actual hardware
DRY_RUN = False  # Log actions without executing

# ======================= PERFORMANCE TUNING =======================
# Threading configuration
MAX_WORKER_THREADS = 4
QUEUE_MAX_SIZE = 100

# Memory limits
MAX_IMAGE_SIZE_MB = 50
MAX_LOG_FILE_SIZE_MB = 100

# ======================= VALIDATION =======================
def validate_config():
    """Validate configuration"""
    errors = []
    
    # Check required directories exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Validate network settings
    if not CONTROL_CENTER_IP:
        errors.append("CONTROL_CENTER_IP is required")
    
    if CLEANING_INTERVAL_DAYS < 1:
        errors.append("CLEANING_INTERVAL_DAYS must be >= 1")
    
    if FAULT_CONFIDENCE_THRESHOLD < 0 or FAULT_CONFIDENCE_THRESHOLD > 1:
        errors.append("FAULT_CONFIDENCE_THRESHOLD must be between 0 and 1")
    
    if errors:
        print("[CONFIG ERROR]")
        for error in errors:
            print(f"  ❌ {error}")
        return False
    
    print("[CONFIG] ✅ Configuration validated successfully")
    return True

if __name__ == "__main__":
    validate_config()
    print("\nConfiguration Summary:")
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Control Center: {CONTROL_CENTER_IP}:{CONTROL_CENTER_PORT}")
    print(f"  Autonomous Mode: {AUTONOMOUS_MODE_ENABLED}")
    print(f"  Cleaning Interval: {CLEANING_INTERVAL_DAYS} days")
    print(f"  Log Path: {LOG_FILE_PATH}")
    print(f"  Reports Path: {MAINTENANCE_REPORT_PATH}")
