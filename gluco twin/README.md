# Gluco Twin — Setup & Requirements

## Python Dependencies (Raspberry Pi)
```
numpy>=1.21
flask>=2.0
pyserial>=3.5
vosk>=0.3.45
pyttsx3>=2.90
sounddevice>=0.4.5
```

Install all:
```bash
pip install numpy flask pyserial vosk pyttsx3 sounddevice
```

## Vosk Model (Offline Indian English STT)
```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip
unzip vosk-model-small-en-in-0.4.zip
# Place the folder next to main.py
```

## Arduino Libraries (install via Arduino IDE → Library Manager)
- SparkFun MAX3010x Pulse and Proximity Sensor Library
- MPU6050 by Electronic Cats
- ArduinoJson by Benoit Blanchon (v6.x)
- heartRate.h + spo2_algorithm.h (included with SparkFun MAX3010x)

## Running the System

### Full system (hardware + voice + dashboard):
```bash
python main.py
```

### Simulation mode (no hardware needed — great for demo backup):
```bash
python main.py --sim
```

### Disable voice (just sensor + dashboard):
```bash
python main.py --no-voice
```

### Test individual modules:
```bash
python digital_twin.py    # test twin model
python voice_handler.py   # test voice pipeline
python sensor_reader.py   # test sensor connection
```

## File Structure
```
gluco_twin/
├── main.py              ← Start everything from here
├── digital_twin.py      ← Kalman filter + physiological model
├── voice_handler.py     ← STT + intent parser + TTS
├── sensor_reader.py     ← Arduino serial reader + sim fallback
├── dashboard.py         ← Flask web dashboard
├── requirements.txt     ← pip install -r requirements.txt
└── arduino/
    └── gluco_twin_sensor/
        └── gluco_twin_sensor.ino   ← Upload to Arduino Nano
```

## Quick Demo Commands (for judges)
Ask the voice assistant:
- "What is my glucose level?"
- "Is my sugar going up?"
- "Am I okay?"
- "What should I eat?"
- "Show me my recent readings"
