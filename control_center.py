"""
SUDO-SOLO Control Center Server
Central server that coordinates between:
- device.cpp (ESP32 motor control) on port 5001
- esp32.c (camera) on port 80
- detect.py (fault detection) on port 5000
- genai.py (report generation)
- automation_pipeline.py (orchestrator)
"""

import os
import sys
import json
import socket
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
from queue import Queue
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_config import (
    CONTROL_CENTER_PORT,
    CONTROL_CENTER_IP,
    LOGS_DIR,
    DEBUG_MODE,
    DEVICE_CONTROL_PORT,
    CAMERA_SERVER_PORT,
    DETECTION_SERVER_PORT,
)

# ======================= LOGGING =======================
def setup_logger():
    """Configure logging"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    logger = logging.getLogger('CONTROL_CENTER')
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    
    fh = logging.FileHandler(os.path.join(LOGS_DIR, 'control_center.log'))
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# ======================= FLASK APP =======================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ======================= SYSTEM STATE =======================
class SystemState:
    """Track system state"""
    
    def __init__(self):
        self.system_mode = "CONTROLLER"  # CONTROLLER or AUTONOMOUS
        self.device_connected = False
        self.camera_ready = False
        self.detector_ready = False
        self.is_cleaning = False
        self.last_cycle_time = None
        self.cycle_count = 0
        self.error_log = []
        
    def to_dict(self):
        return {
            'system_mode': self.system_mode,
            'device_connected': self.device_connected,
            'camera_ready': self.camera_ready,
            'detector_ready': self.detector_ready,
            'is_cleaning': self.is_cleaning,
            'last_cycle_time': self.last_cycle_time,
            'cycle_count': self.cycle_count,
            'timestamp': datetime.now().isoformat(),
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }

system_state = SystemState()

# ======================= DEVICE COMMUNICATION =======================
class DeviceComm:
    """Communicate with ESP32 device controller"""
    
    def __init__(self):
        self.host = CONTROL_CENTER_IP
        self.port = DEVICE_CONTROL_PORT
        self.socket = None
        self.timeout = 5
        
    def connect(self) -> bool:
        """Connect to device"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            system_state.device_connected = True
            logger.info(f"✅ Connected to device at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to device: {e}")
            system_state.device_connected = False
            return False
    
    def send_command(self, command: str) -> bool:
        """Send command to device"""
        try:
            if not self.socket:
                if not self.connect():
                    return False
            
            self.socket.send((command + '\n').encode())
            logger.info(f"📤 Command sent: {command}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to send command: {e}")
            system_state.device_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from device"""
        if self.socket:
            try:
                self.socket.close()
                system_state.device_connected = False
                logger.info("Disconnected from device")
            except:
                pass

device_comm = DeviceComm()

# ======================= FLASK ROUTES =======================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system': system_state.to_dict()
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status"""
    return jsonify(system_state.to_dict()), 200

@app.route('/mode', methods=['GET', 'POST'])
def manage_mode():
    """Get or set system mode"""
    if request.method == 'GET':
        return jsonify({'mode': system_state.system_mode}), 200
    
    elif request.method == 'POST':
        data = request.get_json()
        new_mode = data.get('mode', '').upper()
        
        if new_mode in ['CONTROLLER', 'AUTONOMOUS']:
            system_state.system_mode = new_mode
            logger.info(f"🔀 System mode changed to: {new_mode}")
            return jsonify({
                'message': f'Mode changed to {new_mode}',
                'mode': system_state.system_mode
            }), 200
        else:
            return jsonify({'error': 'Invalid mode. Use CONTROLLER or AUTONOMOUS'}), 400

@app.route('/device/command', methods=['POST'])
def send_device_command():
    """Send command to device controller"""
    try:
        data = request.get_json()
        command = data.get('command', '').upper()
        
        valid_commands = ['F', 'B', 'S', 'START_CLEANING', 'STOP_CLEANING']
        if command not in valid_commands:
            return jsonify({'error': f'Invalid command. Valid: {valid_commands}'}), 400
        
        if device_comm.send_command(command):
            return jsonify({'status': 'sent', 'command': command}), 200
        else:
            return jsonify({'error': 'Failed to send command to device'}), 500
            
    except Exception as e:
        logger.error(f"Command endpoint error: {e}")
        system_state.error_log.append(f"Command error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/device/connect', methods=['POST'])
def connect_device():
    """Connect to device"""
    if device_comm.connect():
        return jsonify({'status': 'connected'}), 200
    else:
        return jsonify({'error': 'Failed to connect to device'}), 500

@app.route('/device/disconnect', methods=['POST'])
def disconnect_device():
    """Disconnect from device"""
    device_comm.disconnect()
    return jsonify({'status': 'disconnected'}), 200

@app.route('/cleaning/start', methods=['POST'])
def start_cleaning():
    """Start cleaning cycle"""
    try:
        data = request.get_json()
        duration = data.get('duration', 120)  # Default 2 minutes
        
        if system_state.is_cleaning:
            return jsonify({'error': 'Cleaning already in progress'}), 400
        
        # Send start command
        if device_comm.send_command('START_CLEANING'):
            system_state.is_cleaning = True
            logger.info(f"🧹 Cleaning started for {duration} seconds")
            return jsonify({
                'status': 'started',
                'duration': duration
            }), 200
        else:
            return jsonify({'error': 'Failed to start cleaning'}), 500
            
    except Exception as e:
        logger.error(f"Start cleaning error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleaning/stop', methods=['POST'])
def stop_cleaning():
    """Stop cleaning cycle"""
    try:
        if device_comm.send_command('STOP_CLEANING'):
            system_state.is_cleaning = False
            logger.info("🛑 Cleaning stopped")
            return jsonify({'status': 'stopped'}), 200
        else:
            return jsonify({'error': 'Failed to stop cleaning'}), 500
            
    except Exception as e:
        logger.error(f"Stop cleaning error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/camera/snapshot', methods=['GET'])
def get_snapshot():
    """Get camera snapshot"""
    try:
        # Forward request to camera server
        import requests
        url = f"http://{CONTROL_CENTER_IP}:{CAMERA_SERVER_PORT}/snapshot"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            system_state.camera_ready = True
            return response.content, 200, {'Content-Type': 'image/jpeg'}
        else:
            system_state.camera_ready = False
            return jsonify({'error': 'Failed to get snapshot'}), 500
            
    except Exception as e:
        logger.error(f"Snapshot error: {e}")
        system_state.camera_ready = False
        system_state.error_log.append(f"Camera error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detection/analyze', methods=['POST'])
def analyze_image():
    """Send image to detection server"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Forward to detection server
        import requests
        url = f"http://{CONTROL_CENTER_IP}:{DETECTION_SERVER_PORT}/detect"
        files = {'image': file.stream}
        response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            system_state.detector_ready = True
            return jsonify(response.json()), 200
        else:
            system_state.detector_ready = False
            return jsonify({'error': 'Detection failed'}), 500
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        system_state.detector_ready = False
        system_state.error_log.append(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/pipeline/run', methods=['POST'])
def run_pipeline():
    """Trigger maintenance pipeline"""
    try:
        if system_state.system_mode != 'AUTONOMOUS':
            return jsonify({'error': 'System not in AUTONOMOUS mode'}), 400
        
        logger.info("📋 Starting automation pipeline...")
        
        # Run pipeline in background
        def run_pipeline_bg():
            try:
                from automation_pipeline import AutomationPipeline
                pipeline = AutomationPipeline()
                pipeline.run_autonomous_maintenance_cycle()
                system_state.cycle_count += 1
                system_state.last_cycle_time = datetime.now().isoformat()
                logger.info(f"✅ Pipeline completed (cycle #{system_state.cycle_count})")
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
                system_state.error_log.append(f"Pipeline error: {e}")
        
        thread = threading.Thread(target=run_pipeline_bg, daemon=True)
        thread.start()
        
        return jsonify({
            'status': 'started',
            'mode': system_state.system_mode
        }), 200
        
    except Exception as e:
        logger.error(f"Pipeline trigger error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """Get recent logs"""
    try:
        limit = request.args.get('limit', 100, type=int)
        log_file = os.path.join(LOGS_DIR, 'control_center.log')
        
        if not os.path.exists(log_file):
            return jsonify({'logs': []}), 200
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        recent_logs = lines[-limit:]
        return jsonify({
            'logs': recent_logs,
            'total_lines': len(lines),
            'returned': len(recent_logs)
        }), 200
        
    except Exception as e:
        logger.error(f"Get logs error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """API endpoint for system status"""
    return jsonify({
        'api_version': 'v1',
        'system': system_state.to_dict()
    }), 200

# ======================= ERROR HANDLERS =======================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ======================= STARTUP =======================
def startup_checks():
    """Perform startup checks"""
    logger.info("=" * 80)
    logger.info("SUDO-SOLO CONTROL CENTER - STARTUP")
    logger.info("=" * 80)
    
    # Check device connectivity
    logger.info("[1/3] Checking device connectivity...")
    if device_comm.connect():
        device_comm.disconnect()
        logger.info("  ✅ Device reachable")
    else:
        logger.warning("  ⚠️  Device not currently reachable (will retry)")
    
    # Check camera
    logger.info("[2/3] Checking camera server...")
    try:
        import requests
        response = requests.get(
            f"http://{CONTROL_CENTER_IP}:{CAMERA_SERVER_PORT}/",
            timeout=5
        )
        system_state.camera_ready = (response.status_code == 200)
        logger.info(f"  {'✅' if system_state.camera_ready else '⚠️'} Camera server {'available' if system_state.camera_ready else 'not available'}")
    except:
        logger.warning("  ⚠️  Camera server not available")
    
    # Check detection server
    logger.info("[3/3] Checking detection server...")
    try:
        response = requests.get(
            f"http://{CONTROL_CENTER_IP}:{DETECTION_SERVER_PORT}/",
            timeout=5
        )
        system_state.detector_ready = (response.status_code in [200, 404])
        logger.info(f"  {'✅' if system_state.detector_ready else '⚠️'} Detection server {'available' if system_state.detector_ready else 'not available'}")
    except:
        logger.warning("  ⚠️  Detection server not available")
    
    logger.info("=" * 80)
    logger.info(f"🚀 Control Center ready on {CONTROL_CENTER_IP}:{CONTROL_CENTER_PORT}")
    logger.info("=" * 80)

# ======================= MAIN =======================
if __name__ == '__main__':
    startup_checks()
    
    logger.info(f"\n📡 Starting Flask server on {CONTROL_CENTER_IP}:{CONTROL_CENTER_PORT}")
    logger.info(f"📚 API Documentation: http://{CONTROL_CENTER_IP}:{CONTROL_CENTER_PORT}/api/v1/status")
    
    # Run Flask app
    app.run(
        host=CONTROL_CENTER_IP,
        port=CONTROL_CENTER_PORT,
        debug=DEBUG_MODE,
        use_reloader=False
    )
