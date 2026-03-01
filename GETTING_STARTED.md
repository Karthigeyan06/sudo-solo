# SUDO-SOLO Automation System - Getting Started Checklist

## Quick Setup (15 minutes)

### Phase 1: Prerequisites Check ✓

- [ ] Python 3.8+ installed
- [ ] pip package manager available
- [ ] 2GB+ available RAM
- [ ] ESP32 device running device.cpp
- [ ] ESP32-CAM device running esp32.c  
- [ ] All on same WiFi network
- [ ] Know your control center IP (e.g., 192.168.1.100)

### Phase 2: Installation

#### Step 1: Install Python Dependencies
```bash
cd d:\sudo-solo
pip install -r requirements_automation.txt
```

Expected output:
```
Successfully installed flask-3.0.0 tensorflow-2.10.0 torch-2.0.0 transformers-4.35.0 ...
```

- [ ] No errors during installation
- [ ] All packages installed successfully

#### Step 2: Configure System
Edit `pipeline_config.py` and update:

```python
# Line 5-7: Update your IP address
CONTROL_CENTER_IP = "192.168.1.100"  # ← CHANGE THIS!
CONTROL_CENTER_PORT = 5000

# Line 20: Enable autonomous mode
AUTONOMOUS_MODE_ENABLED = True

# Line 23: Set cleaning schedule (default 1 week)
CLEANING_INTERVAL_DAYS = 7
```

- [ ] IP address updated to your machine
- [ ] Autonomous mode enabled
- [ ] Cleaning interval set

#### Step 3: Integrate with Hardware (Optional but Recommended)

**For device.cpp:**
1. Open `DEVICE_CPP_INTEGRATION.txt`
2. Copy code blocks into your device.cpp
3. Update IP address constant
4. Compile and upload to ESP32

- [ ] Integration code added to device.cpp
- [ ] IP address updated
- [ ] Compiled and uploaded to device

**For esp32.c:**
1. Open `ESP32_INTEGRATION.txt`
2. Copy code blocks into your esp32.c
3. Update IP address constant
4. Compile and upload to ESP32-CAM

- [ ] Integration code added to esp32.c
- [ ] IP address updated
- [ ] Compiled and uploaded to camera

### Phase 3: Startup

#### Option A: Automatic Startup (Recommended)
```bash
cd d:\sudo-solo
python startup.py
```

This will:
1. Check requirements
2. Validate configuration
3. Check available ports
4. Start all 3 services automatically

- [ ] Startup script ran successfully
- [ ] All 3 components show "started"

#### Option B: Manual Startup (3 Terminal Windows)

**Terminal 1 - Control Center:**
```bash
cd d:\sudo-solo
python control_center.py
```
Expected: `Running on http://192.168.1.100:5000`

- [ ] Control Center running

**Terminal 2 - Detection Server:**
```bash
cd d:\sudo-solo
python detect.py --server
```
Expected: `Starting detection server on 0.0.0.0:5000`

- [ ] Detection server running

**Terminal 3 - Automation Pipeline:**
```bash
cd d:\sudo-solo
python automation_pipeline.py
```
Expected: `Starting scheduler loop...`

- [ ] Pipeline running

### Phase 4: Verification

#### Check Web Dashboard
1. Open browser
2. Go to: `http://192.168.1.100:5000`
3. You should see:
   - System status (mode, devices, etc.)
   - Control panel with buttons
   - Real-time logs

- [ ] Dashboard loads successfully
- [ ] System status shows

#### Test API Endpoints
```bash
# Test 1: Check status
curl http://192.168.1.100:5000/status

# Expected response: JSON with system_mode, device_connected, etc.
```

- [ ] Status endpoint responds
- [ ] Shows system information

#### Test Manual Commands
```bash
# Test 2: Send forward command
curl -X POST http://192.168.1.100:5000/device/command \
  -H "Content-Type: application/json" \
  -d '{"command": "F"}'

# Expected: Device moves forward
```

- [ ] Device responds to commands
- [ ] Robot moves

#### Test Camera
```bash
# Test 3: Get snapshot
curl http://192.168.1.100:5000/camera/snapshot -o test.jpg

# Expected: test.jpg file created with camera image
```

- [ ] Camera responds
- [ ] Image file created

#### Test Full Pipeline
Via dashboard or:
```bash
curl -X POST http://192.168.1.100:5000/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{}'
```

- [ ] Pipeline starts
- [ ] Check logs for progress
- [ ] Report generated in `reports/` folder

### Phase 5: Configuration Review

Review these files:

**Core Configuration:**
- [ ] `pipeline_config.py` - Reviewed and adjusted
- [ ] `AUTOMATION_GUIDE.md` - Read for advanced features
- [ ] Network settings match your hardware

**Thresholds & Behaviors:**
- [ ] Fault confidence threshold (default 0.6)
- [ ] Cleaning duration (default 120 seconds)
- [ ] Schedule interval (default 7 days)

### Phase 6: Testing Workflow

#### Test 1: Manual Cleaning
```
1. Go to dashboard
2. Click "Start Cleaning"
3. Wait 2 minutes
4. Check robot moved
5. Click "Stop Cleaning"
```

- [ ] Cleaning cycle works
- [ ] Duration appropriate
- [ ] Robot properly controlled

#### Test 2: Image Capture
```
1. Dashboard: Click "Capture Image"
2. Check images_captured/ folder
3. Image should show solar panels
```

- [ ] Image saved successfully
- [ ] Image is valid JPEG
- [ ] Quality acceptable

#### Test 3: Fault Detection
```
1. Place a test image in images_captured/
2. Dashboard: Click "Run Full Cycle"
3. Check for fault detection results
4. Review confidence scores
```

- [ ] Detection runs
- [ ] Fault type identified
- [ ] Confidence score shown

#### Test 4: Report Generation
```
1. Check reports/ folder after cycle
2. Review report_*.txt file
3. Verify includes:
   - Fault type
   - Confidence
   - Recommendations
   - Timeline
```

- [ ] Report generated
- [ ] Format correct
- [ ] Content complete

#### Test 5: Scheduling
```
1. Set CLEANING_INTERVAL_DAYS = 0 (temporary)
2. Wait for scheduler check (daily at 6 AM)
   OR manually call pipeline/run
3. Verify cycle executes
4. Set back to 7 days
```

- [ ] Scheduler recognized cycle
- [ ] Pipeline executed
- [ ] Report saved

### Phase 7: Monitoring Setup

#### View Logs
```bash
# Real-time logs
tail -f d:\sudo-solo\logs\automation_pipeline.log

# Last 50 lines
tail -50 d:\sudo-solo\logs\automation_pipeline.log
```

- [ ] Can access logs
- [ ] Logs show system activity

#### Monitor Reports
```bash
# List all reports
dir d:\sudo-solo\reports\

# View latest report
type d:\sudo-solo\reports\report_LATEST.txt
```

- [ ] Reports directory exists
- [ ] Reports storing successfully

### Phase 8: Operational Handoff

#### System Ready When:
- [ ] All components start without errors
- [ ] Dashboard shows system status
- [ ] Manual commands work
- [ ] Camera captures images
- [ ] Fault detection identifies issues
- [ ] Reports generate with recommendations
- [ ] Scheduler is monitoring

#### Recommended Practices:
- [ ] Run manual test cycle first
- [ ] Verify all sensors accessible
- [ ] Review first auto-generated report
- [ ] Monitor 1-2 automated cycles
- [ ] Adjust thresholds if needed

### Phase 9: Production Deployment

For continuous operation:

**Option 1: Windows Task Scheduler**
```
Create task to run: python d:\sudo-solo\startup.py
Trigger: At startup or on schedule
```

**Option 2: Background Service**
```bash
# Using NSSM (Non-Sucking Service Manager)
nssm install SUDO-SOLO-AutomationPipeline "python d:\sudo-solo\startup.py"
nssm start SUDO-SOLO-AutomationPipeline
```

**Option 3: Docker Container**
```bash
docker build -t sudo-solo-automation .
docker run -d --restart always sudo-solo-automation
```

- [ ] Deployment method chosen
- [ ] Service configured
- [ ] Auto-restart enabled

### Phase 10: Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| "Connection refused" | Check IP address, ensure device powered |
| "Model not found" | Verify solar_fault_model.h5 exists |
| "Port in use" | Change port in config or kill process |
| "Pipeline doesn't start" | Check logs, verify all services running |
| "No camera image" | Test ESP32-CAM directly, verify endpoint |
| "Fault detection fails" | Start detect.py server separately |
| "Reports not generating" | Check genai.py, may need to install PyTorch |

- [ ] Understood troubleshooting steps

### Final Verification Checklist

```
SYSTEM STATUS:
□ Control Center running
□ Detection Server running  
□ Automation Pipeline running
□ Dashboard accessible
□ Device responding
□ Camera working
□ Reports generating

CONFIGURATION:
□ IP address correct
□ Autonomous mode enabled
□ Cleaning interval set
□ Thresholds appropriate
□ Logging enabled

OPERATIONS:
□ Logs being written
□ Reports being saved
□ Scheduler monitoring
□ Manual commands working
□ Error handling working
```

### Success Criteria ✓

Your SUDO-SOLO automation system is ready when:

1. ✓ All 3 services start without errors
2. ✓ Dashboard shows live system status
3. ✓ Device responds to movement commands
4. ✓ Camera captures clear images
5. ✓ Faults detected in images
6. ✓ AI-powered reports generated
7. ✓ Reports saved with recommendations
8. ✓ Scheduler monitoring for weekly cycles
9. ✓ Logs recording all activities
10. ✓ System runs 24/7 without intervention

---

## Support Resources

1. **Documentation**
   - `AUTOMATION_GUIDE.md` - Full user guide
   - `IMPLEMENTATION_SUMMARY.md` - Architecture overview
   - `pipeline_config.py` - Config options explained

2. **Integration**
   - `DEVICE_CPP_INTEGRATION.txt` - Hardware code
   - `ESP32_INTEGRATION.txt` - Camera code
   - `API_TESTING.sh` - Test examples

3. **Monitoring**
   - Logs: `logs/automation_pipeline.log`
   - Reports: `reports/report_*.txt`
   - Images: `images_captured/`

4. **Testing**
   - Dashboard: http://192.168.1.100:5000
   - API: http://192.168.1.100:5000/status
   - Logs: `tail -f logs/automation_pipeline.log`

---

## Next Steps After Setup

1. **Customize Thresholds** (pipeline_config.py)
   - Adjust fault confidence threshold
   - Modify cleaning duration
   - Set preferred maintenance schedule

2. **Add Notifications** (pipeline_config.py)
   - Enable email alerts for critical faults
   - Configure alert recipients
   - Set notification thresholds

3. **Integrate Real Sensors** (automation_pipeline.py)
   - Read actual voltage/current sensors
   - Connect temperature sensors
   - Record humidity data

4. **Monitor Performance** (After 1 week)
   - Review all generated reports
   - Check fault detection accuracy
   - Optimize threshold values

5. **Deploy to Production** (Phase 9)
   - Set up system service
   - Configure auto-restart
   - Enable remote monitoring

---

**Congratulations!** Your SUDO-SOLO Automation System is now operational.

For advanced configuration and troubleshooting, see `AUTOMATION_GUIDE.md`

---

**Last Updated:** March 2024  
**Status:** Ready to Use  
**Support:** Refer to AUTOMATION_GUIDE.md for detailed documentation
