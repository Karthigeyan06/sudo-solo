"""
SUDO-SOLO Automation Pipeline
Main orchestrator for autonomous solar panel cleaning and fault detection
Control Flow: Mode Check → Autonomous Cleaning → Image Capture → Fault Detection → AI Report Generation
"""

import os
import sys
import json
import time
import logging
import threading
import schedule
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import socket

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ======================= CONFIGURATION =======================
from pipeline_config import (
    DEVICE_CONTROL_PORT,
    CAMERA_SERVER_PORT,
    DETECTION_SERVER_PORT,
    CONTROL_CENTER_IP,
    CONTROL_CENTER_PORT,
    AUTONOMOUS_MODE_ENABLED,
    CLEANING_INTERVAL_DAYS,
    MAINTENANCE_REPORT_PATH,
    LOG_FILE_PATH,
)

# ======================= LOGGER SETUP =======================
def setup_logger(name, log_file=LOG_FILE_PATH):
    """Configure logging for the automation pipeline"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger('AUTOMATION_PIPELINE')

# ======================= DATA STRUCTURES =======================
@dataclass
class FaultDetectionResult:
    """Result from fault detection"""
    timestamp: str
    fault_type: str
    confidence: float
    image_path: str
    probabilities: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class MaintenanceReport:
    """Maintenance report structure"""
    timestamp: str
    session_id: str
    robot_mode: str
    cleaning_performed: bool
    fault_detected: bool
    fault_type: Optional[str]
    fault_confidence: Optional[float]
    ai_report: str
    sensor_data: Dict
    status: str  # 'success', 'partial', 'failed'

# ======================= SYSTEM STATUS CHECK =======================
class SystemHealthMonitor:
    """Monitor system health and connectivity"""
    
    def __init__(self):
        self.logger = setup_logger('SYSTEM_HEALTH')
        
    def check_device_connectivity(self, ip: str, port: int, timeout: int = 2) -> bool:
        """Check if device is reachable"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception as e:
            self.logger.error(f"Connectivity check failed for {ip}:{port}: {e}")
            return False
    
    def check_camera_ready(self) -> bool:
        """Check if camera server is running"""
        return self.check_device_connectivity(CONTROL_CENTER_IP, CAMERA_SERVER_PORT)
    
    def check_detection_server_ready(self) -> bool:
        """Check if detection server is running"""
        return self.check_device_connectivity(CONTROL_CENTER_IP, DETECTION_SERVER_PORT)
    
    def check_all_systems(self) -> Dict[str, bool]:
        """Check all system components"""
        status = {
            'camera': self.check_camera_ready(),
            'detection': self.check_detection_server_ready(),
            'timestamp': datetime.now().isoformat()
        }
        self.logger.info(f"System health check: {status}")
        return status

# ======================= DEVICE CONTROL =======================
class DeviceController:
    """Control robot movement and sensors"""
    
    def __init__(self, control_ip: str = CONTROL_CENTER_IP, control_port: int = CONTROL_CENTER_PORT):
        self.control_ip = control_ip
        self.control_port = control_port
        self.logger = setup_logger('DEVICE_CONTROL')
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to device control server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.control_ip, self.control_port))
            self.is_connected = True
            self.logger.info(f"Connected to device at {self.control_ip}:{self.control_port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to device: {e}")
            self.is_connected = False
            return False
    
    def send_command(self, command: str) -> bool:
        """Send command to device"""
        if not self.is_connected:
            if not self.connect():
                return False
        
        try:
            self.socket.send((command + '\n').encode())
            self.logger.info(f"Sent command: {command}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            self.is_connected = False
            return False
    
    def start_cleaning_sequence(self, duration_seconds: int = 60) -> bool:
        """Execute cleaning sequence (forward movement with brush)"""
        self.logger.info(f"Starting cleaning sequence for {duration_seconds} seconds")
        if self.send_command("START_CLEANING"):
            time.sleep(duration_seconds)
            return self.send_command("STOP_CLEANING")
        return False
    
    def move_forward(self, duration: int = 10) -> bool:
        """Move forward"""
        if self.send_command("F"):
            time.sleep(duration)
            return self.send_command("S")
        return False
    
    def stop(self) -> bool:
        """Stop all movement"""
        return self.send_command("S")
    
    def disconnect(self):
        """Disconnect from device"""
        if self.is_connected:
            try:
                self.socket.close()
                self.is_connected = False
                self.logger.info("Disconnected from device")
            except:
                pass

# ======================= CAMERA & IMAGE CAPTURE =======================
class CameraController:
    """Control camera and image capture"""
    
    def __init__(self, camera_url: str = f"http://{CONTROL_CENTER_IP}:{CAMERA_SERVER_PORT}"):
        self.camera_url = camera_url
        self.logger = setup_logger('CAMERA_CONTROL')
        self.image_cache_dir = os.path.join(os.path.dirname(__file__), 'images_captured')
        os.makedirs(self.image_cache_dir, exist_ok=True)
    
    def capture_image(self, filename: Optional[str] = None) -> Optional[str]:
        """Capture image from camera"""
        try:
            now = datetime.now()
            if filename is None:
                filename = f"solar_panel_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            
            image_path = os.path.join(self.image_cache_dir, filename)
            
            # Try to get image from camera stream
            response = requests.get(f"{self.camera_url}/snapshot", timeout=10)
            
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                self.logger.info(f"Image captured: {image_path}")
                return image_path
            else:
                self.logger.error(f"Failed to capture image: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Image capture failed: {e}")
            return None
    
    def get_latest_image(self) -> Optional[str]:
        """Get the most recently captured image"""
        try:
            images = sorted(
                Path(self.image_cache_dir).glob('*.jpg'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if images:
                return str(images[0])
        except Exception as e:
            self.logger.error(f"Failed to get latest image: {e}")
        return None

# ======================= FAULT DETECTION =======================
class FaultDetectionService:
    """Interface with fault detection server"""
    
    def __init__(self, detection_url: str = f"http://{CONTROL_CENTER_IP}:{DETECTION_SERVER_PORT}"):
        self.detection_url = detection_url
        self.logger = setup_logger('FAULT_DETECTION')
    
    def detect_fault(self, image_path: str) -> Optional[FaultDetectionResult]:
        """Detect fault in image"""
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.detection_url}/detect",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                result = FaultDetectionResult(
                    timestamp=datetime.now().isoformat(),
                    fault_type=data.get('label', 'Unknown'),
                    confidence=max(data.get('probabilities', [0])),
                    image_path=image_path,
                    probabilities={
                        'Burn': data['probabilities'][0] if len(data['probabilities']) > 0 else 0,
                        'Crack': data['probabilities'][1] if len(data['probabilities']) > 1 else 0,
                        'Delamination': data['probabilities'][2] if len(data['probabilities']) > 2 else 0,
                        'Dust': data['probabilities'][3] if len(data['probabilities']) > 3 else 0,
                        'Normal': data['probabilities'][4] if len(data['probabilities']) > 4 else 0,
                    }
                )
                self.logger.info(f"Fault detected: {result.fault_type} ({result.confidence:.2%})")
                return result
            else:
                self.logger.error(f"Detection failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Fault detection error: {e}")
            return None

# ======================= AI REPORT GENERATION =======================
class ReportGenerator:
    """Generate AI maintenance reports"""
    
    def __init__(self):
        self.logger = setup_logger('REPORT_GENERATION')
        
    def generate_report(self, 
                       fault_detection: FaultDetectionResult,
                       sensor_data: Dict) -> str:
        """Generate maintenance report using GenAI"""
        try:
            # Build prompt for AI report generation
            prompt = self._create_prompt(fault_detection, sensor_data)
            
            # Import and use genai module
            from genai import generate_with_local_llm, generate_fallback_report
            
            report = generate_with_local_llm(prompt)
            
            if report:
                self.logger.info("Report generated successfully with AI")
                return report
            else:
                self.logger.warning("AI report failed, using fallback template")
                return self._generate_fallback_report(fault_detection, sensor_data)
                
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return self._generate_fallback_report(fault_detection, sensor_data)
    
    def _create_prompt(self, fault_detection: FaultDetectionResult, sensor_data: Dict) -> str:
        """Create structured prompt for AI"""
        return f"""You are a solar panel maintenance expert. Create a detailed maintenance report.

DETECTED FAULT: {fault_detection.fault_type}
CONFIDENCE LEVEL: {fault_detection.confidence:.1%}

PROBABILITIES:
- Burn: {fault_detection.probabilities.get('Burn', 0):.2%}
- Crack: {fault_detection.probabilities.get('Crack', 0):.2%}
- Delamination: {fault_detection.probabilities.get('Delamination', 0):.2%}
- Dust: {fault_detection.probabilities.get('Dust', 0):.2%}
- Normal: {fault_detection.probabilities.get('Normal', 0):.2%}

SENSOR READINGS:
- Voltage: {sensor_data.get('Voltage', 'N/A')} V
- Current: {sensor_data.get('Current', 'N/A')} A
- Temperature: {sensor_data.get('Temperature', 'N/A')} °C
- Humidity: {sensor_data.get('Humidity', 'N/A')} %
- Pressure: {sensor_data.get('Pressure', 'N/A')} hPa

REPORT STRUCTURE:
1. FAULT EXPLANATION: Explain the detected fault
2. SEVERITY: Low/Medium/High assessment
3. IMPACT: Effects on panel efficiency
4. PREVENTIVE ACTIONS: 3-5 specific actions
5. CORRECTIVE STEPS: Immediate maintenance steps
6. TIMELINE: Recommended action timeline

Be professional and concise."""

    def _generate_fallback_report(self, fault_detection: FaultDetectionResult, sensor_data: Dict) -> str:
        """Generate fallback report when AI is unavailable"""
        severity = "HIGH" if fault_detection.confidence > 0.8 else "MEDIUM" if fault_detection.confidence > 0.6 else "LOW"
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════════════╗
║          SOLAR PANEL MAINTENANCE REPORT - AUTOMATED GENERATION                 ║
╚════════════════════════════════════════════════════════════════════════════════╝

REPORT DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
FAULT DETECTED: {fault_detection.fault_type}
CONFIDENCE: {fault_detection.confidence:.2%}
SEVERITY: {severity}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FAULT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Detected Fault Type: {fault_detection.fault_type}
   Detection Confidence: {fault_detection.confidence:.2%}
   
   Fault Breakdown:
   • Burn Damage: {fault_detection.probabilities.get('Burn', 0):.2%}
   • Crack Damage: {fault_detection.probabilities.get('Crack', 0):.2%}
   • Delamination: {fault_detection.probabilities.get('Delamination', 0):.2%}
   • Dust Accumulation: {fault_detection.probabilities.get('Dust', 0):.2%}
   • Normal Condition: {fault_detection.probabilities.get('Normal', 0):.2%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CURRENT SENSOR READINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Voltage: {sensor_data.get('Voltage', 'N/A')} V
   Current: {sensor_data.get('Current', 'N/A')} A
   Temperature: {sensor_data.get('Temperature', 'N/A')} °C
   Humidity: {sensor_data.get('Humidity', 'N/A')} %
   Pressure: {sensor_data.get('Pressure', 'N/A')} hPa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. SEVERITY ASSESSMENT: {severity}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Impact on Efficiency: {self._estimate_efficiency_loss(fault_detection.fault_type)}% reduction
   Status: {'REQUIRES IMMEDIATE ATTENTION' if severity == 'HIGH' else 'Schedule maintenance within 1 week' if severity == 'MEDIUM' else 'Monitor and plan preventive action'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. PREVENTIVE ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✓ Establish regular cleaning schedule (every 2-4 weeks)
   ✓ Install weather-resistant panel covers
   ✓ Implement dust monitoring system
   ✓ Apply anti-soiling coatings
   ✓ Trim surrounding vegetation and clear obstructions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. CORRECTIVE ACTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   1. Perform thorough cleaning with soft-bristle brush
   2. Use deionized water for rinsing (avoid tap water)
   3. Test panel output immediately after cleaning
   4. Document maintenance action in system log
   5. Schedule follow-up inspection in 1 month

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. RECOMMENDED TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Immediate Action Required: {"YES - URGENT" if severity == "HIGH" else "NO - Schedule within 2 weeks"}
   Follow-up Inspection: 30 days
   Next Preventive Maintenance: 60 days
   System Monitoring: Continuous

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORT GENERATED BY: SUDO-SOLO Autonomous Maintenance System
IMAGE ANALYZED: {fault_detection.image_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return report
    
    def _estimate_efficiency_loss(self, fault_type: str) -> int:
        """Estimate efficiency loss by fault type"""
        loss_map = {
            'Dust': 15,
            'Crack': 25,
            'Burn': 35,
            'Delamination': 30,
            'Normal': 0
        }
        return loss_map.get(fault_type, 10)

# ======================= AUTOMATION ORCHESTRATOR =======================
class AutomationPipeline:
    """Main automation orchestrator"""
    
    def __init__(self):
        self.logger = setup_logger('AUTOMATION_ORCHESTRATOR')
        self.device = DeviceController()
        self.camera = CameraController()
        self.detector = FaultDetectionService()
        self.reporter = ReportGenerator()
        self.health = SystemHealthMonitor()
        self.last_cleaning = None
        self.session_id = None
        
        # Mock sensor data (integrate with actual sensors)
        self.sensor_data = {
            'Voltage': 17.8,
            'Current': 4.3,
            'Temperature': 42,
            'Humidity': 65,
            'Pressure': 1012
        }
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def check_system_mode(self) -> str:
        """Check if system is in autonomous or controller mode"""
        # In a real implementation, this would query the device
        return "AUTONOMOUS" if AUTONOMOUS_MODE_ENABLED else "CONTROLLER"
    
    def execute_cleaning_cycle(self) -> bool:
        """Execute cleaning cycle"""
        self.logger.info("🤖 STARTING CLEANING CYCLE")
        try:
            if not self.device.connect():
                self.logger.error("Failed to connect to device")
                return False
            
            # Send cleaning command
            success = self.device.start_cleaning_sequence(duration_seconds=120)
            
            if success:
                self.last_cleaning = datetime.now()
                self.logger.info("✅ Cleaning cycle completed successfully")
                return True
            else:
                self.logger.error("❌ Cleaning cycle failed")
                return False
                
        finally:
            self.device.disconnect()
    
    def execute_detection_cycle(self) -> Optional[FaultDetectionResult]:
        """Execute fault detection cycle"""
        self.logger.info("📸 STARTING FAULT DETECTION CYCLE")
        try:
            # Capture image
            image_path = self.camera.capture_image()
            if not image_path:
                self.logger.error("Failed to capture image")
                return None
            
            # Detect fault
            result = self.detector.detect_fault(image_path)
            if result:
                self.logger.info(f"✅ Fault detection completed: {result.fault_type}")
                return result
            else:
                self.logger.error("❌ Fault detection failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Detection cycle error: {e}")
            return None
    
    def execute_report_generation(self, fault_detection: FaultDetectionResult) -> str:
        """Generate maintenance report"""
        self.logger.info("📄 GENERATING MAINTENANCE REPORT")
        try:
            report = self.reporter.generate_report(fault_detection, self.sensor_data)
            self.logger.info("✅ Report generated successfully")
            return report
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            return ""
    
    def run_autonomous_maintenance_cycle(self) -> MaintenanceReport:
        """Execute complete autonomous maintenance cycle"""
        self.session_id = self.generate_session_id()
        start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 AUTONOMOUS MAINTENANCE CYCLE STARTED - {self.session_id}")
        self.logger.info("=" * 80)
        
        report = MaintenanceReport(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            robot_mode="AUTONOMOUS",
            cleaning_performed=False,
            fault_detected=False,
            fault_type=None,
            fault_confidence=None,
            ai_report="",
            sensor_data=self.sensor_data,
            status="in_progress"
        )
        
        try:
            # Step 1: Check system health
            self.logger.info("\n[Step 1/4] Checking system health...")
            health_status = self.health.check_all_systems()
            if not health_status.get('camera') or not health_status.get('detection'):
                self.logger.warning("⚠️  Not all systems ready, proceeding with available systems")
            
            # Step 2: Execute cleaning
            self.logger.info("\n[Step 2/4] Executing cleaning sequence...")
            cleaning_success = self.execute_cleaning_cycle()
            report.cleaning_performed = cleaning_success
            
            # Step 3: Detect faults
            self.logger.info("\n[Step 3/4] Capturing image and detecting faults...")
            fault_detection = self.execute_detection_cycle()
            
            if fault_detection:
                report.fault_detected = True
                report.fault_type = fault_detection.fault_type
                report.fault_confidence = fault_detection.confidence
                
                # Step 4: Generate report
                self.logger.info("\n[Step 4/4] Generating AI maintenance report...")
                ai_report = self.execute_report_generation(fault_detection)
                report.ai_report = ai_report
            else:
                self.logger.warning("No fault detection result available")
            
            report.status = "success"
            
        except Exception as e:
            self.logger.error(f"Cycle execution error: {e}")
            report.status = "failed"
        
        # Save report
        self.save_report(report)
        
        duration = (datetime.now() - start_time).total_seconds()
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"✅ CYCLE COMPLETED - Status: {report.status.upper()} ({duration:.1f}s)")
        self.logger.info("=" * 80)
        
        return report
    
    def save_report(self, report: MaintenanceReport):
        """Save report to file"""
        try:
            os.makedirs(os.path.dirname(MAINTENANCE_REPORT_PATH), exist_ok=True)
            
            report_file = os.path.join(
                MAINTENANCE_REPORT_PATH,
                f"report_{report.session_id}.json"
            )
            
            with open(report_file, 'w') as f:
                json.dump({
                    'session_id': report.session_id,
                    'timestamp': report.timestamp,
                    'robot_mode': report.robot_mode,
                    'cleaning_performed': report.cleaning_performed,
                    'fault_detected': report.fault_detected,
                    'fault_type': report.fault_type,
                    'fault_confidence': report.fault_confidence,
                    'sensor_data': report.sensor_data,
                    'status': report.status,
                    'ai_report': report.ai_report
                }, f, indent=2)
            
            self.logger.info(f"Report saved: {report_file}")
            
            # Also save as text file
            report_text_file = report_file.replace('.json', '.txt')
            with open(report_text_file, 'w') as f:
                f.write(report.ai_report)
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
    
    def should_run_cleaning(self) -> bool:
        """Check if weekly cleaning should run"""
        if self.last_cleaning is None:
            return True  # First run
        
        days_since = (datetime.now() - self.last_cleaning).days
        return days_since >= CLEANING_INTERVAL_DAYS
    
    def schedule_maintenance(self):
        """Schedule weekly maintenance"""
        def job():
            if self.should_run_cleaning():
                self.run_autonomous_maintenance_cycle()
            else:
                days_left = CLEANING_INTERVAL_DAYS - (datetime.now() - self.last_cleaning).days
                self.logger.info(f"Next cleaning scheduled in {days_left} days")
        
        # Schedule for daily check at 6 AM
        schedule.every().day.at("06:00").do(job)
        
        self.logger.info(f"📅 Weekly cleaning scheduled (every {CLEANING_INTERVAL_DAYS} days)")
    
    def run_scheduler(self):
        """Run the scheduler in continuous loop"""
        self.logger.info("🔄 Starting scheduler loop...")
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)

# ======================= MAIN ENTRY POINT =======================
def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("SUDO-SOLO AUTOMATION PIPELINE - INITIALIZATION")
    logger.info("=" * 80)
    
    # Create pipeline
    pipeline = AutomationPipeline()
    
    # Check system mode
    mode = pipeline.check_system_mode()
    logger.info(f"System Mode: {mode}")
    
    if mode == "AUTONOMOUS":
        logger.info("System in AUTONOMOUS MODE - Starting automation pipeline")
        
        # Option 1: Run single maintenance cycle
        # pipeline.run_autonomous_maintenance_cycle()
        
        # Option 2: Schedule weekly maintenance (recommended for production)
        pipeline.schedule_maintenance()
        pipeline.run_scheduler()
    else:
        logger.info("System in CONTROLLER MODE - Automation disabled")
        logger.info("Switch to AUTONOMOUS MODE to enable scheduling")

if __name__ == "__main__":
    main()
