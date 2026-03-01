#define BLYNK_TEMPLATE_ID "TMPL3903N5g02"
#define BLYNK_TEMPLATE_NAME "solo"
#define BLYNK_PRINT Serial

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

/* ------------------- CREDENTIALS ------------------- */
char auth[] = "YOUR_BLYNK_AUTH";
char ssid[] = "YOUR_WIFI";
char pass[] = "YOUR_PASS";

/* ------------------- MOTOR PINS ------------------- */
#define IN1 25
#define IN2 26
#define IN3 27
#define IN4 14
#define ENA 32
#define ENB 33
#define LED_PIN 2

/* ------------------- PWM CONFIG ------------------- */
#define PWM_CHANNEL_A 0
#define PWM_CHANNEL_B 1
#define PWM_FREQ 1000
#define PWM_RESOLUTION 8

int speedPWM = 150;

/* ------------------- SYSTEM MODES ------------------- */
enum SystemMode { CONTROLLER_MODE, AUTONOMOUS_MODE };
SystemMode currentMode = CONTROLLER_MODE;

/* ------------------- CONTROL CENTER CONFIG ------------------- */
WiFiClient controlClient;
const char* controlIP = "192.168.1.100";
const uint16_t controlPort = 5000;
unsigned long lastReconnectAttempt = 0;

/* ------------------- LED BLINK ------------------- */
unsigned long lastBlink = 0;
bool ledState = false;

/* ================================================== */
/* ================== MOTOR CONTROL ================= */
/* ================================================== */

void moveForward()
{
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveBackward()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void stopMotor()
{
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

/* ================================================== */
/* ================== BLYNK HANDLERS ================ */
/* ================================================== */

// Speed slider V2
BLYNK_WRITE(V2)
{
  if (currentMode != CONTROLLER_MODE) return;

  speedPWM = map(param.asInt(), 0, 1024, 0, 255);
  ledcWrite(PWM_CHANNEL_A, speedPWM);
  ledcWrite(PWM_CHANNEL_B, speedPWM);
}

// Forward button V0
BLYNK_WRITE(V0)
{
  if (currentMode != CONTROLLER_MODE) return;

  if (param.asInt())
    moveForward();
  else
    stopMotor();
}

// Backward button V1
BLYNK_WRITE(V1)
{
  if (currentMode != CONTROLLER_MODE) return;

  if (param.asInt())
    moveBackward();
  else
    stopMotor();
}

// Mode switch V5
BLYNK_WRITE(V5)
{
  if (param.asInt() == 0)
  {
    currentMode = CONTROLLER_MODE;
    Serial.println("Switched to CONTROLLER MODE");

    if (controlClient.connected())
      controlClient.stop();
  }
  else
  {
    currentMode = AUTONOMOUS_MODE;
    Serial.println("Switched to AUTONOMOUS MODE");
  }
}

/* ================================================== */
/* ================== AUTONOMOUS MODE =============== */
/* ================================================== */

void handleAutonomousMode()
{
  if (!controlClient.connected())
  {
    if (millis() - lastReconnectAttempt > 3000)
    {
      Serial.println("Connecting to Control Center...");
      controlClient.connect(controlIP, controlPort);
      lastReconnectAttempt = millis();
    }
    return;
  }

  while (controlClient.available())
  {
    String command = controlClient.readStringUntil('\n');
    command.trim();

    if (command == "F")
      moveForward();
    else if (command == "B")
      moveBackward();
    else if (command == "S")
      stopMotor();
  }
}

/* ================================================== */
/* ================== POWER LED ===================== */
/* ================================================== */

void blinkPowerLED()
{
  if (millis() - lastBlink > 500)
  {
    ledState = !ledState;
    digitalWrite(LED_PIN, ledState);
    lastBlink = millis();
  }
}

/* ================================================== */
/* ================== SETUP ========================= */
/* ================================================== */

void setup()
{
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(LED_PIN, OUTPUT);

  // PWM setup
  ledcSetup(PWM_CHANNEL_A, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(ENA, PWM_CHANNEL_A);

  ledcSetup(PWM_CHANNEL_B, PWM_FREQ, PWM_RESOLUTION);
  ledcAttachPin(ENB, PWM_CHANNEL_B);

  stopMotor();

  Blynk.begin(auth, ssid, pass);
}

/* ================================================== */
/* ================== LOOP ========================== */
/* ================================================== */

void loop()
{
  if (currentMode == CONTROLLER_MODE)
  {
    Blynk.run();
  }
  else
  {
    handleAutonomousMode();
  }

  blinkPowerLED();
}