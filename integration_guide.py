"""
INTEGRATION GUIDE for device.cpp and esp32.c
Add these modifications to enable automation system communication
"""

DEVICE_CPP_MODIFICATIONS = """
// ============================================================================
// INTEGRATION: Add this to device.cpp for Automation Pipeline Support
// ============================================================================

// Add this at the TOP of device.cpp file (after existing #includes):

#include <WiFiClient.h>
#include <vector>

/* ================== AUTOMATION SYSTEM CONFIG ================== */
const char* CONTROL_CENTER_IP = "192.168.1.100";  // ❗ UPDATE THIS WITH YOUR CONTROL CENTER IP
const uint16_t CONTROL_CENTER_PORT = 5001;

/* ================== CONTROL CLIENT ================== */
WiFiClient automationClient;
bool automationConnected = false;
unsigned long lastAutomationReconnect = 0;
const unsigned long AUTOMATION_RECONNECT_INTERVAL = 5000;  // Try every 5 seconds

/* ================== AUTONOMOUS MODE CLEANING ================== */
bool isCleaningActive = false;
unsigned long cleaningStartTime = 0;
unsigned long cleaningDuration = 120000;  // 2 minutes in milliseconds

/* ================== FUNCTION DEFINITIONS ================== */

/* Connect to Automation Control Center */
void connectToAutomationCenter() {
    if (automationConnected) return;
    
    Serial.print("[AUTOMATION] Connecting to Control Center at ");
    Serial.print(CONTROL_CENTER_IP);
    Serial.print(":");
    Serial.println(CONTROL_CENTER_PORT);
    
    if (automationClient.connect(CONTROL_CENTER_IP, CONTROL_CENTER_PORT)) {
        automationConnected = true;
        Serial.println("[AUTOMATION] ✅ Connected to Control Center!");
        automationClient.println("STATUS:CONNECTED");
    } else {
        automationConnected = false;
        Serial.println("[AUTOMATION] ❌ Failed to connect to Control Center");
    }
}

/* Disconnect from Automation Control Center */
void disconnectFromAutomationCenter() {
    if (automationClient.connected()) {
        automationClient.stop();
        automationConnected = false;
        Serial.println("[AUTOMATION] Disconnected from Control Center");
    }
}

/* Handle commands from Automation System */
void handleAutomationCommands() {
    if (!automationClient.available()) return;
    
    String command = automationClient.readStringUntil('\n');
    command.trim();
    
    Serial.print("[AUTOMATION] Received command: ");
    Serial.println(command);
    
    if (command == "F") {
        moveForward();
    } 
    else if (command == "B") {
        moveBackward();
    } 
    else if (command == "S") {
        stopMotor();
    } 
    else if (command == "START_CLEANING") {
        startCleaningCycle();
    } 
    else if (command == "STOP_CLEANING") {
        stopCleaningCycle();
    }
    else if (command == "STATUS") {
        automationClient.println("STATUS:READY");
    }
    else if (command == "PING") {
        automationClient.println("PONG");
    }
}

/* Start cleaning cycle (autonomous cleaning) */
void startCleaningCycle() {
    if (currentMode != AUTONOMOUS_MODE) {
        Serial.println("[CLEANING] ❌ Not in autonomous mode!");
        return;
    }
    
    if (isCleaningActive) {
        Serial.println("[CLEANING] ⚠️  Cleaning already in progress");
        return;
    }
    
    Serial.println("[CLEANING] 🧹 Starting cleaning cycle...");
    Serial.println("[CLEANING] Enabling brush motor and moving forward");
    
    isCleaningActive = true;
    cleaningStartTime = millis();
    
    // Move forward to perform cleaning
    moveForward();
    
    // Enable brush motor (assuming GPIO pin for brush)
    // digitalWrite(BRUSH_MOTOR_PIN, HIGH);  // Uncomment if you have brush motor
    
    automationClient.println("STATUS:CLEANING_STARTED");
}

/* Stop cleaning cycle */
void stopCleaningCycle() {
    if (!isCleaningActive) {
        Serial.println("[CLEANING] ⚠️  No cleaning in progress");
        return;
    }
    
    Serial.println("[CLEANING] 🛑 Stopping cleaning cycle");
    
    // Stop motors
    stopMotor();
    
    // Disable brush motor if enabled
    // digitalWrite(BRUSH_MOTOR_PIN, LOW);  // Uncomment if you have brush motor
    
    isCleaningActive = false;
    automationClient.println("STATUS:CLEANING_STOPPED");
}

/* Monitor cleaning duration and auto-stop */
void monitorCleaningCycle() {
    if (!isCleaningActive) return;
    
    unsigned long elapsed = millis() - cleaningStartTime;
    
    if (elapsed >= cleaningDuration) {
        Serial.print("[CLEANING] Duration reached (");
        Serial.print(elapsed / 1000);
        Serial.println("s), stopping cleaning");
        stopCleaningCycle();
    }
}

/* Maintain Automation connection */
void maintainAutomationConnection() {
    if (currentMode == AUTONOMOUS_MODE) {
        if (!automationConnected || !automationClient.connected()) {
            if (millis() - lastAutomationReconnect > AUTOMATION_RECONNECT_INTERVAL) {
                connectToAutomationCenter();
                lastAutomationReconnect = millis();
            }
        } else {
            handleAutomationCommands();
        }
    }
}

// ============================================================================
// MODIFICATION: Add this to your existing setup() function
// ============================================================================

void setup() {
    // ... existing setup code ...
    
    // Add this at the end of setup():
    if (currentMode == AUTONOMOUS_MODE) {
        Serial.println("[AUTOMATION] System in AUTONOMOUS mode - enabling Control Center connection");
        connectToAutomationCenter();
    }
}

// ============================================================================
// MODIFICATION: Update your existing loop() function
// ============================================================================

void loop() {
    if (currentMode == CONTROLLER_MODE) {
        Blynk.run();
    } 
    else {
        // AUTONOMOUS MODE
        maintainAutomationConnection();
        handleAutonomousMode();
        monitorCleaningCycle();
    }
    
    blinkPowerLED();
}

// ============================================================================
// OPTIONAL: Update mode switch handler for better integration
// ============================================================================

// Replace existing BLYNK_WRITE(V5) with this:
BLYNK_WRITE(V5) {
    if (param.asInt() == 0) {
        currentMode = CONTROLLER_MODE;
        Serial.println("Switched to CONTROLLER MODE");
        
        if (automationClient.connected()) {
            disconnectFromAutomationCenter();
        }
        if (controlClient.connected()) {
            controlClient.stop();
        }
    }
    else {
        currentMode = AUTONOMOUS_MODE;
        Serial.println("Switched to AUTONOMOUS MODE");
        connectToAutomationCenter();
    }
}

// ============================================================================
// OPTIONAL: Add this function to enable/disable brush motor (if available)
// ============================================================================

#define BRUSH_MOTOR_PIN 12  // Adjust pin number as needed

void enableBrushMotor() {
    digitalWrite(BRUSH_MOTOR_PIN, HIGH);
    Serial.println("[BRUSH] Motor enabled");
}

void disableBrushMotor() {
    digitalWrite(BRUSH_MOTOR_PIN, LOW);
    Serial.println("[BRUSH] Motor disabled");
}

// Uncomment this in setup() if using brush motor:
// pinMode(BRUSH_MOTOR_PIN, OUTPUT);
// disableBrushMotor();  // Start disabled

"""

ESP32_MODIFICATIONS = """
// ============================================================================
// INTEGRATION: Add this to esp32.c for Automation Pipeline Support
// ============================================================================

// Add this at the TOP of esp32.c file (after existing #includes):

#include <HTTPClient.h>
#include <Wire.h>

/* ================== AUTOMATION SYSTEM CONFIG ================== */
const char* CONTROL_CENTER_IP = "192.168.1.100";  // ❗ UPDATE THIS WITH YOUR CONTROL CENTER IP
const int CONTROL_CENTER_PORT = 5000;
const char* SNAPSHOT_ENDPOINT = "/camera/snapshot";
const char* UPLOAD_ENDPOINT = "/detection/analyze";

/* ================== CAPTURE SCHEDULING ================== */
unsigned long lastCaptureTime = 0;
const unsigned long CAPTURE_INTERVAL = 3600000;  // 1 hour in milliseconds
uint32_t captureCount = 0;

/* ================== SERVER ENDPOINTS ================== */

// Add this endpoint to serve snapshots to Control Center
static esp_err_t snapshot_handler(httpd_req_t *req){
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    
    fb = esp_camera_fb_get();
    if (!fb) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Camera capture failed");
        return ESP_FAIL;
    }
    
    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=snapshot.jpg");
    
    if (fb->format == PIXFORMAT_JPEG) {
        res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    } else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Non-JPEG format");
        res = ESP_FAIL;
    }
    
    esp_camera_fb_return(fb);
    
    Serial.print("[SNAPSHOT] Captured and sent to Control Center (");
    Serial.print(res == ESP_OK ? "✅" : "❌");
    Serial.println(")");
    
    return res;
}

// Add URI for snapshot endpoint
static httpd_uri_t snapshot_uri = {
    .uri       = "/snapshot",
    .method    = HTTP_GET,
    .handler   = snapshot_handler,
    .user_ctx  = NULL
};

// ============================================================================
// AUTO-UPLOAD FUNCTIONS
// ============================================================================

/* Capture and upload image to Control Center for analysis */
bool captureAndUploadImage() {
    camera_fb_t * fb = NULL;
    
    Serial.println("[UPLOAD] Capturing image for analysis...");
    
    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[UPLOAD] ❌ Failed to capture frame");
        return false;
    }
    
    // Create HTTP client
    HTTPClient http;
    String uploadURL = "http://" + String(CONTROL_CENTER_IP) + ":" + String(CONTROL_CENTER_PORT) + UPLOAD_ENDPOINT;
    
    Serial.print("[UPLOAD] Uploading to: ");
    Serial.println(uploadURL);
    
    // Start HTTP POST request
    http.begin(uploadURL);
    http.addHeader("Content-Type", "image/jpeg");
    
    // Send image data
    int httpResponseCode = http.POST(fb->buf, fb->len);
    
    if (httpResponseCode > 0) {
        String response = http.getString();
        Serial.print("[UPLOAD] ✅ Response code: ");
        Serial.println(httpResponseCode);
        Serial.print("[UPLOAD] Response: ");
        Serial.println(response);
    } else {
        Serial.print("[UPLOAD] ❌ Error code: ");
        Serial.println(httpResponseCode);
    }
    
    http.end();
    esp_camera_fb_return(fb);
    
    captureCount++;
    return (httpResponseCode > 0 && httpResponseCode < 300);
}

/* Send status to Control Center */
void sendStatusToCenterCenter() {
    HTTPClient http;
    String statusURL = "http://" + String(CONTROL_CENTER_IP) + ":" + String(CONTROL_CENTER_PORT) + "/status";
    
    http.begin(statusURL);
    http.addHeader("Content-Type", "application/json");
    
    String payload = "{\"camera_status\":\"ready\",\"captures\":" + String(captureCount) + "}";
    int httpResponseCode = http.POST(payload);
    
    if (httpResponseCode > 0) {
        Serial.print("[STATUS] Sent to Control Center (HTTP ");
        Serial.print(httpResponseCode);
        Serial.println(")");
    }
    
    http.end();
}

/* Check Control Center for capture request */
bool checkForCaptureRequest() {
    HTTPClient http;
    String cmdURL = "http://" + String(CONTROL_CENTER_IP) + ":" + String(CONTROL_CENTER_PORT) + "/camera/capture";
    
    http.begin(cmdURL);
    int httpResponseCode = http.GET();
    
    bool shouldCapture = (httpResponseCode == 200);
    
    if (shouldCapture) {
        Serial.println("[COMMAND] Received capture request from Control Center");
    }
    
    http.end();
    return shouldCapture;
}

// ============================================================================
// MODIFICATION: Add to your setup() function
// ============================================================================

void setup() {
    // ... existing setup code ...
    
    // Add this at the end of setup():
    
    // Register snapshot endpoint
    httpd_register_uri_handler(stream_httpd, &snapshot_uri);
    
    Serial.println("[AUTOMATION] Camera system ready for automation pipeline");
    Serial.print("[AUTOMATION] Control Center: ");
    Serial.print(CONTROL_CENTER_IP);
    Serial.print(":");
    Serial.println(CONTROL_CENTER_PORT);
}

// ============================================================================
// MODIFICATION: Add this to your loop() function
// ============================================================================

void loop() {
    // ... existing loop code ...
    
    // Add periodic communications with Control Center
    unsigned long now = millis();
    
    // Send status every 30 seconds
    static unsigned long lastStatusTime = 0;
    if (now - lastStatusTime > 30000) {
        sendStatusToCenterCenter();
        lastStatusTime = now;
    }
    
    // Check for capture requests from Control Center every 5 seconds
    static unsigned long lastCommandCheck = 0;
    if (now - lastCommandCheck > 5000) {
        if (checkForCaptureRequest()) {
            captureAndUploadImage();
        }
        lastCommandCheck = now;
    }
}

// ============================================================================
// OPTIONAL: Auto-capture on schedule (hourly)
// ============================================================================

void monitorAutoCapture() {
    unsigned long now = millis();
    
    if (now - lastCaptureTime > CAPTURE_INTERVAL) {
        Serial.println("[SCHEDULER] Auto-capture triggered");
        if (captureAndUploadImage()) {
            Serial.println("[SCHEDULER] ✅ Auto-capture successful");
        } else {
            Serial.println("[SCHEDULER] ❌ Auto-capture failed");
        }
        lastCaptureTime = now;
    }
}

// Add this to loop() as well:
    monitorAutoCapture();

"""

API_TESTING = """
// ============================================================================
// TESTING THE INTEGRATION
// ============================================================================

# Test 1: Check device connectivity
curl -X POST http://192.168.1.100:5000/device/connect

# Test 2: Send command to device
curl -X POST http://192.168.1.100:5000/device/command \\
  -H "Content-Type: application/json" \\
  -d '{"command": "F"}'

# Test 3: Get camera snapshot
curl http://192.168.1.100:5000/camera/snapshot -o snapshot.jpg

# Test 4: Start cleaning
curl -X POST http://192.168.1.100:5000/cleaning/start \\
  -H "Content-Type: application/json" \\
  -d '{"duration": 120}'

# Test 5: Check system status
curl http://192.168.1.100:5000/status

# Test 6: Switch to autonomous mode
curl -X POST http://192.168.1.100:5000/mode \\
  -H "Content-Type: application/json" \\
  -d '{"mode": "AUTONOMOUS"}'

# Test 7: Run full pipeline
curl -X POST http://192.168.1.100:5000/pipeline/run \\
  -H "Content-Type: application/json" \\
  -d '{}'

"""

if __name__ == "__main__":
    import os
    
    # Create integration guide file
    output_dir = os.path.dirname(__file__)
    
    with open(os.path.join(output_dir, 'DEVICE_CPP_INTEGRATION.txt'), 'w', encoding='utf-8') as f:
        f.write(DEVICE_CPP_MODIFICATIONS)
    
    with open(os.path.join(output_dir, 'ESP32_INTEGRATION.txt'), 'w', encoding='utf-8') as f:
        f.write(ESP32_MODIFICATIONS)
    
    with open(os.path.join(output_dir, 'API_TESTING.sh'), 'w', encoding='utf-8') as f:
        f.write(API_TESTING)
    
    print("Integration guides created:")
    print("  - DEVICE_CPP_INTEGRATION.txt")
    print("  - ESP32_INTEGRATION.txt")
    print("  - API_TESTING.sh")
