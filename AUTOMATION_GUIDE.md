# 🤖 SUDO-SOLO Automation Pipeline Guide

## System Overview

The automation pipeline orchestrates an autonomous solar panel maintenance system that:

1. **Checks System Mode** - Distinguishes between CONTROLLER and AUTONOMOUS modes
2. **Performs Weekly Cleaning** - Uses device.cpp to control robot movement for cleaning
3. **Captures Images** - Uses esp32.c camera to take photos of solar panels
4. **Detects Faults** - Uses detect.py to identify panel damage/defects
5. **Generates Reports** - Uses genai.py to create AI-powered maintenance reports

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUDO-SOLO CONTROL CENTER                     │
│          (Orchestrates all subsystems & scheduling)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │ DEVICE   │  │ CAMERA   │  │ DETECTOR │
          │(device.c │  │(esp32.c) │  │(detect.py)
          │pp)       │  │          │  │          │
          └──────────┘  └──────────┘  └──────────┘
                │             │             │
                └─────────────▼─────────────┘
                              │
                        ┌─────▼──────┐
                        │ REPORT GEN │
                        │ (genai.py)  │
                        └─────┬──────┘
                              │
                        ┌─────▼──────────┐
                        │MAINTENANCE REP.│
                        └────────────────┘
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_automation.txt
```

### 2. Configure System

Edit `pipeline_config.py`:

```python
CONTROL_CENTER_IP = "YOUR_MACHINE_IP"  # Update this!
AUTONOMOUS_MODE_ENABLED = True
CLEANING_INTERVAL_DAYS = 7  # Weekly
```

### 3. Start Control Center

```bash
# Terminal 1: Start Control Center
python control_center.py
```

### 4. Start Detection Server

```bash
# Terminal 2: Start fault detection server
python detect.py --server
```

### 5. Start Automation Pipeline

```bash
# Terminal 3: Start automation scheduler
python automation_pipeline.py
```

### 6. Access Dashboard

Open browser and go to:
```
http://YOUR_MACHINE_IP:5000
```

---

## Component Integration

### Device Control (device.cpp)

The robot uses an ESP32 with motor control. Integration points:

1. **Connection**: TCP socket on port 5001
2. **Commands**:
   - `F` - Move forward
   - `B` - Move backward
   - `S` - Stop
   - `START_CLEANING` - Begin autonomous cleaning cycle
   - `STOP_CLEANING` - End cleaning cycle

**Example Hardware Integration:**
```cpp
#define CONTROL_CENTER_IP "192.168.1.100"
#define CONTROL_CENTER_PORT 5001

void connectToControlCenter() {
  controlClient.connect(CONTROL_CENTER_IP, CONTROL_CENTER_PORT);
}

void receiveCommand() {
  while (controlClient.available()) {
    String command = controlClient.readStringUntil('\n');
    command.trim();
    
    if (command == "START_CLEANING") {
      // Enable brush motor
      moveForward();
      // Continue for 2 minutes
    }
  }
}
```

### Camera System (esp32.c)

The camera captures images of solar panels:

1. **Endpoint**: `/snapshot` on port 80
2. **Output**: JPEG image sent via HTTP
3. **Resolution**: Configurable (default 640x480)
4. **Quality**: 80% JPEG compression

**Example Web Server:**
```c
static esp_err_t snapshot_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) {
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
  }
  return ESP_OK;
}
```

### Fault Detection (detect.py)

Pre-configured to accept images and return fault predictions:

```python
# Already running on port 5000
# Endpoint: POST /detect
# Input: Image file
# Output: {"label": "Dust", "probabilities": [...]}
```

### Report Generation (genai.py)

Uses local LLM for AI-powered maintenance reports:

```python
# Gemma-2B or Phi-2 model for report generation
# Fallback template if LLM unavailable
# Includes: fault explanation, severity, actions, timeline
```

---

## API Reference

All endpoints use JSON and are available at `http://CONTROL_CENTER_IP:5000`

### System Status

**GET** `/status`
```json
{
  "system_mode": "AUTONOMOUS",
  "device_connected": true,
  "camera_ready": true,
  "detector_ready": true,
  "is_cleaning": false,
  "cycle_count": 5,
  "last_cycle_time": "2024-03-01T14:30:00"
}
```

### Mode Management

**GET** `/mode`
```json
{"mode": "AUTONOMOUS"}
```

**POST** `/mode`
```json
{"mode": "AUTONOMOUS"}  // or "CONTROLLER"
```

### Device Control

**POST** `/device/command`
```json
{"command": "F"}  // "F", "B", "S", "START_CLEANING", "STOP_CLEANING"
```

### Cleaning Operations

**POST** `/cleaning/start`
```json
{"duration": 120}  // seconds
```

**POST** `/cleaning/stop`
```json
{}
```

### Pipeline Execution

**POST** `/pipeline/run`
```
Triggers complete autonomous cycle:
1. Health check
2. Cleaning
3. Image capture
4. Fault detection
5. Report generation
```

### Logs

**GET** `/logs?limit=100`
```json
{
  "logs": ["2024-03-01 14:30:00 - [INFO] - Starting cycle...", ...],
  "total_lines": 1250,
  "returned": 100
}
```

---

## Configuration Examples

### Weekly Cleaning Schedule

```python
# pipeline_config.py
CLEANING_INTERVAL_DAYS = 7
MAINTENANCE_CHECK_HOUR = 6  # Check at 6 AM
```

The system will automatically clean weekly on the scheduled day and time.

### Autonomous Mode with Fault Alerts

```python
# pipeline_config.py
AUTONOMOUS_MODE_ENABLED = True
FAULT_CONFIDENCE_THRESHOLD = 0.6
CRITICAL_FAULTS = ['Burn', 'Delamination']
ENABLE_EMAIL_NOTIFICATIONS = True
```

### Manual One-Time Cycle

```python
from automation_pipeline import AutomationPipeline

pipeline = AutomationPipeline()
report = pipeline.run_autonomous_maintenance_cycle()
```

---

## Maintenance Reports

### Report Structure

Each maintenance report includes:

1. **Fault Analysis**
   - Detected fault type
   - Confidence level
   - Probability breakdown

2. **Sensor Readings**
   - Voltage, current, temperature
   - Humidity, pressure

3. **Severity Assessment**
   - Critical/High/Medium/Low classification
   - Efficiency impact estimation

4. **Action Items**
   - Preventive measures
   - Corrective steps
   - Recommended timeline

5. **AI Insights**
   - Generated by local LLM
   - Actionable recommendations
   - Best practices

### Report Output

Reports are saved in:
- `reports/report_SESSION_TIMESTAMP.json` - Structured data
- `reports/report_SESSION_TIMESTAMP.txt` - Human-readable format

---

## Troubleshooting

### Device Not Connecting

```
Error: Failed to connect to device at 192.168.1.100:5001
```

**Solution:**
1. Check device.cpp is running on ESP32
2. Verify IP address in pipeline_config.py
3. Ensure device is powered and connected to WiFi

### Camera Snapshot Fails

```
Error: Failed to get snapshot
```

**Solution:**
1. Verify esp32.c is running on camera device
2. Check camera port configuration (default 80)
3. Test camera directly: `curl http://192.168.1.X/snapshot`

### Detection Server Not Available

```
Error: Detection server not available
```

**Solution:**
1. Start detection server: `python detect.py --server`
2. Verify port 5000 is not in use
3. Check firewall settings

### LLM Report Generation Fails

```
Warning: Using fallback report (LLM failed)
```

**Solution:**
1. Install PyTorch: `pip install torch`
2. Download model: `pip install transformers`
3. Increase available RAM/VRAM
4. System will use template fallback if LLM unavailable

### Scheduler Not Running

**Solution:**
1. Ensure automation_pipeline.py is running
2. Check system clock is correct
3. Verify CLEANING_INTERVAL_DAYS >= 1
4. Check logs: `python logs`

---

## Performance Tuning

### For Faster Cycles

```python
# pipeline_config.py
CLEANING_DURATION_SECONDS = 60  # Reduce from 120
FAULT_CONFIDENCE_THRESHOLD = 0.7  # Higher threshold, fewer false positives
```

### For Higher Accuracy

```python
# pipeline_config.py
CLEANING_DURATION_SECONDS = 180  # More thorough cleaning
FAULT_CONFIDENCE_THRESHOLD = 0.5  # Lower threshold, catch more issues
LLM_TEMPERATURE = 0.5  # More consistent outputs
```

### For Lower Memory Usage

```python
# pipeline_config.py
LLM_MODEL = "google/gemma-2b"  # Lighter than Phi-2
LLM_MAX_TOKENS = 200  # Shorter reports
```

---

## Running in Production

### Systemd Service (Linux/Raspberry Pi)

Create `/etc/systemd/system/sudo-solo.service`:

```ini
[Unit]
Description=SUDO-SOLO Automation Pipeline
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/sudo-solo
ExecStart=/usr/bin/python3 /home/pi/sudo-solo/automation_pipeline.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable sudo-solo
sudo systemctl start sudo-solo
```

### Docker Deployment

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements_automation.txt .
RUN pip install -r requirements_automation.txt
COPY . .
CMD ["python", "automation_pipeline.py"]
```

---

## Advanced Features

### Custom Sensor Integration

```python
# In automation_pipeline.py
class AutomationPipeline:
    def read_sensors(self):
        # Connect to your sensor hardware
        voltage = read_voltage_sensor()
        current = read_current_sensor()
        temp = read_temperature_sensor()
        
        self.sensor_data = {
            'Voltage': voltage,
            'Current': current,
            'Temperature': temp,
            # Add more sensors...
        }
```

### Custom Cleaning Patterns

```python
# In device_comm or automation_pipeline.py
def execute_grid_cleaning(self, rows=2, cols=3):
    """Move in grid pattern across panel"""
    for row in range(rows):
        for col in range(cols):
            move_to_position(row, col)
            start_brush()
            time.sleep(30)
            stop_brush()
```

### Conditional Maintenance

```python
# Auto-schedule maintenance based on fault severity
fault = pipeline.execute_detection_cycle()
if fault.confidence > 0.8:  # High confidence
    schedule_immediate_maintenance()
elif fault.confidence > 0.6:  # Medium confidence
    schedule_week_maintenance()
else:
    schedule_monthly_checkup()
```

---

## Support & Debugging

### Enable Debug Mode

```python
# pipeline_config.py
DEBUG_MODE = True
LOG_LEVEL = 'DEBUG'
```

### View Real-time Logs

```bash
tail -f logs/automation_pipeline.log
```

### Check System Health

```python
python -c "from automation_pipeline import SystemHealthMonitor; health = SystemHealthMonitor(); print(health.check_all_systems())"
```

### Test Single Components

```bash
# Test camera
python -c "from automation_pipeline import CameraController; c = CameraController(); print(c.capture_image())"

# Test detection
python -c "from automation_pipeline import FaultDetectionService; d = FaultDetectionService(); print(d.detect_fault('image.jpg'))"

# Test device
python -c "from automation_pipeline import DeviceController; d = DeviceController(); d.connect(); d.send_command('F')"
```

---

## License & Disclaimer

This automation system is designed for controlled environments. Ensure:
- ✅ Proper electrical safety measures
- ✅ Hardware properly grounded and isolated
- ✅ Testing in safe environment before deployment
- ✅ Regular maintenance and inspections
- ✅ Emergency stop mechanisms functional

---

**Version:** 1.0  
**Last Updated:** March 2024  
**Maintained by:** SUDO-SOLO Development Team
