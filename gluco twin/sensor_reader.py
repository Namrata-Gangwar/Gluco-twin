"""
Gluco Twin — Sensor Reader
Reads sensor data from Arduino Nano over USB serial (/dev/ttyUSB0)
and feeds it into the GlucoTwin digital twin model.

Arduino sends JSON every 500ms:
{"ir":18420,"red":19010,"hr":72.3,"spo2":98.1,"ax":0.1,"ay":-0.05,"az":9.82,"temp":33.7,"ecg":510}
"""

import serial
import json
import time
import threading
from typing import Optional, Callable
from digital_twin import SensorReading, GlucoTwin, GlucoTwinSimulator

# ─── Arduino Serial Reader ────────────────────────────────────────────────────

class ArduinoSensorReader:
    """
    Connects to Arduino Nano over USB serial and parses JSON sensor packets.
    Falls back to simulation if no Arduino is found.
    """

    BAUD_RATE    = 115200
    READ_TIMEOUT = 2.0  # seconds

    def __init__(self, port: str = "/dev/ttyUSB0",
                 on_reading: Optional[Callable[[SensorReading], None]] = None):
        self.port       = port
        self.on_reading = on_reading
        self._serial    = None
        self._running   = False
        self._thread    = None
        self._last_raw  = {}

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(
                self.port,
                self.BAUD_RATE,
                timeout=self.READ_TIMEOUT
            )
            time.sleep(2)   # wait for Arduino reset
            print(f"[SERIAL] Connected to Arduino on {self.port} @ {self.BAUD_RATE} baud")
            return True
        except serial.SerialException as e:
            print(f"[SERIAL] Could not connect to {self.port}: {e}")
            return False

    def disconnect(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[SERIAL] Disconnected.")

    def _parse_line(self, line: str) -> Optional[SensorReading]:
        """Parse one JSON line from Arduino into a SensorReading."""
        try:
            d = json.loads(line.strip())
            self._last_raw = d
            return SensorReading(
                timestamp  = time.time(),
                ppg_ir     = int(d.get("ir",    18000)),
                ppg_red    = int(d.get("red",   19000)),
                heart_rate = float(d.get("hr",  70.0)),
                spo2       = float(d.get("spo2", 98.0)),
                accel_x    = float(d.get("ax",  0.0)),
                accel_y    = float(d.get("ay",  0.0)),
                accel_z    = float(d.get("az",  9.81)),
                skin_temp  = float(d.get("temp", 33.5)),
                ecg_value  = float(d.get("ecg", 512.0)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Could be a debug print from Arduino — skip silently
            return None

    def read_loop(self):
        """Blocking read loop. Call in a thread."""
        self._running = True
        while self._running:
            try:
                raw_line = self._serial.readline().decode("utf-8", errors="ignore")
                if raw_line.startswith("{"):
                    reading = self._parse_line(raw_line)
                    if reading and self.on_reading:
                        self.on_reading(reading)
                elif raw_line.strip():
                    print(f"[ARDUINO DEBUG] {raw_line.strip()}")
            except serial.SerialException as e:
                print(f"[SERIAL] Read error: {e}")
                break

    def start(self):
        if not self._serial or not self._serial.is_open:
            if not self.connect():
                print("[SERIAL] Starting in simulation mode.")
                return False
        self._thread = threading.Thread(target=self.read_loop, daemon=True)
        self._thread.start()
        return True

    @property
    def last_raw(self) -> dict:
        return self._last_raw

# ─── Sensor Manager ──────────────────────────────────────────────────────────

class SensorManager:
    """
    High-level manager that:
    - Tries to connect to Arduino
    - Falls back to simulation automatically
    - Feeds readings into GlucoTwin
    - Provides get_latest() for Flask + voice handler
    """

    def __init__(self, port: str = "/dev/ttyUSB0"):
        self.twin          = GlucoTwin(initial_glucose=100.0)
        self._latest       = None
        self._latest_lock  = threading.Lock()
        self._sim          = None
        self._sim_thread   = None
        self._reader       = ArduinoSensorReader(
            port=port,
            on_reading=self._handle_reading
        )
        self._mode         = "unknown"

    def _handle_reading(self, reading: SensorReading):
        estimate = self.twin.ingest(reading)
        with self._latest_lock:
            self._latest = estimate
        if estimate.alert:
            print(f"[ALERT] {estimate.alert}")

    def start(self):
        success = self._reader.start()
        if success:
            self._mode = "hardware"
            print("[SENSOR] Running in HARDWARE mode.")
        else:
            self._mode = "simulation"
            print("[SENSOR] Running in SIMULATION mode.")
            self._sim = GlucoTwinSimulator()
            self._sim.twin = self.twin   # share the twin instance
            self._sim_thread = threading.Thread(
                target=self._sim_loop, daemon=True
            )
            self._sim_thread.start()

    def _sim_loop(self):
        while True:
            estimate = self._sim.step()
            with self._latest_lock:
                self._latest = estimate
            time.sleep(1.0)

    def get_latest(self) -> Optional[dict]:
        with self._latest_lock:
            return self.twin.get_summary()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def raw_sensor(self) -> dict:
        if self._mode == "hardware":
            return self._reader.last_raw
        return {}


# ─── Test ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Sensor Manager (auto-detects hardware or simulation)...")
    manager = SensorManager(port="/dev/ttyUSB0")
    manager.start()

    print(f"\nMode: {manager.mode}\nReading for 15 seconds...\n")
    for i in range(15):
        time.sleep(1)
        data = manager.get_latest()
        if data and data["glucose"]:
            print(f"[{i+1:2d}s] Glucose: {data['glucose']:6.1f} mg/dL  "
                  f"Trend: {data['trend']:20s}  "
                  f"{'ALERT: ' + data['alert'] if data['alert'] else 'OK'}")
