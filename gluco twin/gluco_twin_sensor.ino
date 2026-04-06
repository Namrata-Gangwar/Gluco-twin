/*
  Gluco Twin — Arduino Nano Sensor Sketch
  Reads MAX30102, MPU-6050, LM35, AD8232
  Sends JSON over Serial at 115200 baud every 500ms

  Libraries (install via Arduino Library Manager):
    - SparkFun MAX3010x Pulse and Proximity Sensor Library
    - MPU6050 by Electronic Cats  (or Adafruit MPU6050)
    - Wire (built-in)
    - ArduinoJson  v6.x

  Wiring:
    MAX30102  → Nano A4(SDA), A5(SCL), 3.3V, GND
    MPU-6050  → Nano A4(SDA), A5(SCL), 3.3V, GND, AD0→GND
    LM35      → Nano A0, 5V, GND
    AD8232    → Nano A1(OUTPUT), D2(LO+), D3(LO-)
    Buzzer    → Nano D9 via 100Ω
    LED-G     → Nano D10 via 220Ω
    LED-R     → Nano D11 via 220Ω
*/

#include <Wire.h>
#include "MAX30105.h"           // SparkFun MAX3010x library
#include "heartRate.h"          // BPM algorithm from SparkFun
#include "spo2_algorithm.h"     // SpO2 algorithm from SparkFun
#include <MPU6050.h>            // Electronic Cats MPU6050
#include <ArduinoJson.h>

// ── Pin definitions ─────────────────────────────────────────────────────────
#define PIN_LM35        A0     // LM35 temperature sensor
#define PIN_AD8232      A1     // AD8232 ECG output
#define PIN_LO_PLUS     2      // AD8232 leads-off detection +
#define PIN_LO_MINUS    3      // AD8232 leads-off detection -
#define PIN_BUZZER      9      // Active buzzer
#define PIN_LED_GREEN   10     // Green LED (normal)
#define PIN_LED_RED     11     // Red LED (alert)

// ── Objects ──────────────────────────────────────────────────────────────────
MAX30105  particleSensor;
MPU6050   mpu;

// ── BPM / SpO2 state (SparkFun algorithm) ───────────────────────────────────
#define BUFFER_SIZE     100
uint32_t  irBuffer[BUFFER_SIZE],  redBuffer[BUFFER_SIZE];
int32_t   bufferLength;
int32_t   spo2Value;
int8_t    validSPO2;
int32_t   heartRateValue;
int8_t    validHeartRate;

// ── Simple moving-average BPM ────────────────────────────────────────────────
const byte RATE_SIZE = 4;
byte       rates[RATE_SIZE];
byte       rateSpot = 0;
long       lastBeat = 0;
float      beatsPerMinute;
int        beatAvg;

// ── Timing ───────────────────────────────────────────────────────────────────
unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL_MS = 500;

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Wire.begin();

  pinMode(PIN_LO_PLUS,  INPUT);
  pinMode(PIN_LO_MINUS, INPUT);
  pinMode(PIN_BUZZER,   OUTPUT);
  pinMode(PIN_LED_GREEN, OUTPUT);
  pinMode(PIN_LED_RED,   OUTPUT);

  // MAX30102 init
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("{\"error\":\"MAX30102 not found\"}");
    while (1);   // halt — check wiring
  }
  particleSensor.setup();
  particleSensor.setPulseAmplitudeRed(0x0A);   // low power
  particleSensor.setPulseAmplitudeGreen(0);    // off

  // MPU-6050 init
  mpu.initialize();
  if (!mpu.testConnection()) {
    Serial.println("{\"error\":\"MPU6050 not found\"}");
  }

  // Pre-fill SpO2 buffer
  bufferLength = BUFFER_SIZE;
  for (int i = 0; i < bufferLength; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();
  }
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, bufferLength, redBuffer,
    &spo2Value, &validSPO2,
    &heartRateValue, &validHeartRate
  );

  digitalWrite(PIN_LED_GREEN, HIGH);  // ready indicator
  delay(100);
  digitalWrite(PIN_LED_GREEN, LOW);
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // ── Shift SpO2 buffer and read new samples ──────────────────────────────
  for (int i = 25; i < BUFFER_SIZE; i++) {
    redBuffer[i - 25] = redBuffer[i];
    irBuffer[i - 25]  = irBuffer[i];
  }
  for (int i = 75; i < BUFFER_SIZE; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i]  = particleSensor.getIR();
    particleSensor.nextSample();

    // Simple BPM detection alongside SpO2 algorithm
    if (checkForBeat(irBuffer[i])) {
      long delta  = millis() - lastBeat;
      lastBeat    = millis();
      beatsPerMinute = 60 / (delta / 1000.0);
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        beatAvg = 0;
        for (int j = 0; j < RATE_SIZE; j++) beatAvg += rates[j];
        beatAvg /= RATE_SIZE;
      }
    }
  }

  // ── Recalculate SpO2 ────────────────────────────────────────────────────
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, bufferLength, redBuffer,
    &spo2Value, &validSPO2,
    &heartRateValue, &validHeartRate
  );

  // ── Read MPU-6050 ────────────────────────────────────────────────────────
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  float ax_g = ax / 16384.0;  // convert to g (±2g range)
  float ay_g = ay / 16384.0;
  float az_g = az / 16384.0;

  // ── Read LM35 (temperature) ─────────────────────────────────────────────
  int   tempRaw  = analogRead(PIN_LM35);
  float tempC    = (tempRaw * 5.0 / 1023.0) * 100.0;  // LM35: 10mV/°C

  // ── Read AD8232 (ECG) ───────────────────────────────────────────────────
  int ecgValue = 0;
  if (digitalRead(PIN_LO_PLUS) == 0 && digitalRead(PIN_LO_MINUS) == 0) {
    ecgValue = analogRead(PIN_AD8232);
  }

  // ── Send JSON every SEND_INTERVAL_MS ────────────────────────────────────
  if (millis() - lastSend >= SEND_INTERVAL_MS) {
    lastSend = millis();

    StaticJsonDocument<256> doc;
    doc["ir"]   = irBuffer[BUFFER_SIZE - 1];
    doc["red"]  = redBuffer[BUFFER_SIZE - 1];
    doc["hr"]   = validHeartRate ? heartRateValue : beatAvg;
    doc["spo2"] = validSPO2     ? spo2Value       : 98;
    doc["ax"]   = serialized(String(ax_g * 9.81, 3));   // convert g → m/s²
    doc["ay"]   = serialized(String(ay_g * 9.81, 3));
    doc["az"]   = serialized(String(az_g * 9.81, 3));
    doc["temp"] = serialized(String(tempC, 1));
    doc["ecg"]  = ecgValue;
    doc["ms"]   = millis();

    serializeJson(doc, Serial);
    Serial.println();   // newline so Pi can readline()
  }

  // ── LED indicators (based on latest ir signal strength) ─────────────────
  long irVal = irBuffer[BUFFER_SIZE - 1];
  if (irVal < 5000) {
    // No finger detected
    digitalWrite(PIN_LED_GREEN, LOW);
    digitalWrite(PIN_LED_RED,   LOW);
  } else {
    digitalWrite(PIN_LED_GREEN, HIGH);
    digitalWrite(PIN_LED_RED,   LOW);
  }
}
