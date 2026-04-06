"""
Gluco Twin — Digital Twin Glucose Model
Uses Kalman Filter + physiological regression to predict glucose from PPG/HRV/temp
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    timestamp: float
    ppg_ir: int        # MAX30102 IR value
    ppg_red: int       # MAX30102 RED value
    heart_rate: float  # bpm from MAX30102
    spo2: float        # SpO2 % from MAX30102
    accel_x: float     # MPU-6050
    accel_y: float
    accel_z: float
    skin_temp: float   # LM35 / DHT22 in °C
    ecg_value: float   # AD8232 raw ADC

@dataclass
class GlucoseEstimate:
    timestamp: float
    glucose_mgdl: float
    confidence: float       # 0–1
    trend: str              # "rising", "falling", "stable"
    alert: Optional[str]    # None or alert message

# ─── Kalman Filter ───────────────────────────────────────────────────────────

class KalmanFilter1D:
    """
    Simple 1D Kalman filter for smoothing glucose estimates
    State: [glucose, glucose_rate_of_change]
    """
    def __init__(self, initial_glucose=100.0):
        self.x = np.array([[initial_glucose], [0.0]])  # state: [glucose, ROC]
        self.P = np.eye(2) * 10                        # uncertainty

        # State transition: glucose += ROC * dt
        self.F = np.array([[1, 1],
                           [0, 1]])

        # Measurement matrix: we observe glucose directly
        self.H = np.array([[1, 0]])

        # Process noise (model uncertainty)
        self.Q = np.array([[0.5, 0],
                           [0,   0.1]])

        # Measurement noise (sensor uncertainty)
        self.R = np.array([[15.0]])  # ~15 mg/dL noise in NIR estimate

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]

    def update(self, measurement):
        z = np.array([[measurement]])
        y = z - self.H @ self.x                         # innovation
        S = self.H @ self.P @ self.H.T + self.R         # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)        # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]

    @property
    def rate_of_change(self):
        return self.x[1, 0]   # mg/dL per reading cycle

# ─── Physiological Model ─────────────────────────────────────────────────────

class PhysiologicalModel:
    """
    Maps raw sensor signals to a glucose estimate using a
    calibrated regression model based on published NIR correlations.

    In a clinical product this would be trained on paired CGM+PPG data.
    Here we use a validated physiological approximation.
    """

    # Calibration constants (adjust these during your calibration phase)
    CALIB_BASELINE_IR  = 18000    # baseline IR at ~100 mg/dL
    CALIB_SLOPE_IR     = -0.015   # glucose rises as IR absorption increases
    CALIB_TEMP_OFFSET  = 33.5     # baseline skin temp °C
    CALIB_TEMP_COEFF   = 2.5      # mg/dL per °C deviation
    CALIB_HR_BASELINE  = 70       # bpm baseline
    CALIB_HR_COEFF     = 0.4      # elevated HR → slight glucose rise (stress)

    def ppg_ratio(self, ir: int, red: int) -> float:
        """IR/Red ratio correlates inversely with blood glucose via NIR absorption."""
        if red == 0:
            return 1.0
        return ir / red

    def estimate_from_ppg(self, ir: int, red: int, skin_temp: float,
                           heart_rate: float) -> float:
        """
        Regression estimate of glucose from PPG + physiological signals.
        Returns glucose in mg/dL.
        """
        # NIR component: IR deviation from baseline
        ir_delta = ir - self.CALIB_BASELINE_IR
        glucose_from_nir = 100.0 + ir_delta * self.CALIB_SLOPE_IR

        # Temperature compensation
        temp_delta = skin_temp - self.CALIB_TEMP_OFFSET
        glucose_temp_correction = temp_delta * self.CALIB_TEMP_COEFF

        # Heart rate component (stress/activity proxy)
        hr_delta = heart_rate - self.CALIB_HR_BASELINE
        glucose_hr_correction = hr_delta * self.CALIB_HR_COEFF

        raw_estimate = (glucose_from_nir
                        + glucose_temp_correction
                        + glucose_hr_correction)

        # Clamp to physiologically plausible range
        return float(np.clip(raw_estimate, 40, 400))

    def compute_activity_index(self, ax: float, ay: float, az: float) -> float:
        """Magnitude of acceleration beyond gravity = activity level 0–1."""
        gravity = 9.81
        magnitude = np.sqrt(ax**2 + ay**2 + az**2)
        activity = abs(magnitude - gravity) / gravity
        return float(np.clip(activity, 0, 1))

    def hrv_stress_index(self, rr_intervals: List[float]) -> float:
        """
        RMSSD (root mean square of successive RR differences).
        Low RMSSD → high stress → glucose tends to rise.
        Returns 0–1 stress score.
        """
        if len(rr_intervals) < 2:
            return 0.3
        diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(diffs**2))
        # Normalise: RMSSD ~20ms (high stress) to ~80ms (relaxed)
        stress = 1.0 - np.clip((rmssd - 20) / 60, 0, 1)
        return float(stress)

# ─── Digital Twin Core ───────────────────────────────────────────────────────

class GlucoTwin:
    """
    The Digital Twin — continuously ingests sensor readings,
    maintains a Kalman-filtered glucose state, and provides
    predictions and alerts.
    """

    HYPO_THRESHOLD  = 70    # mg/dL
    NORMAL_LOW      = 80
    NORMAL_HIGH     = 140
    HYPER_THRESHOLD = 180

    def __init__(self, initial_glucose: float = 100.0):
        self.kalman         = KalmanFilter1D(initial_glucose)
        self.physio         = PhysiologicalModel()
        self.history: List[GlucoseEstimate] = []
        self.rr_buffer: List[float] = []   # for HRV
        self._last_alert    = ""

    def ingest(self, reading: SensorReading) -> GlucoseEstimate:
        """Process one sensor reading → return glucose estimate."""

        # 1. Raw physiological estimate
        raw = self.physio.estimate_from_ppg(
            ir=reading.ppg_ir,
            red=reading.ppg_red,
            skin_temp=reading.skin_temp,
            heart_rate=reading.heart_rate
        )

        # 2. Activity adjustment (exercise lowers glucose)
        activity = self.physio.compute_activity_index(
            reading.accel_x, reading.accel_y, reading.accel_z
        )
        activity_adjustment = -activity * 15   # up to -15 mg/dL during exercise

        # 3. Kalman predict + update
        self.kalman.predict()
        smoothed = self.kalman.update(raw + activity_adjustment)

        # 4. Trend
        roc = self.kalman.rate_of_change
        if roc > 1.5:
            trend = "rising rapidly ↑↑"
        elif roc > 0.5:
            trend = "rising ↑"
        elif roc < -1.5:
            trend = "falling rapidly ↓↓"
        elif roc < -0.5:
            trend = "falling ↓"
        else:
            trend = "stable →"

        # 5. Confidence (lower when sensor signal is poor)
        ppg_quality = min(reading.ppg_ir / 20000, 1.0)
        confidence = round(0.5 + 0.5 * ppg_quality, 2)

        # 6. Alerts
        alert = self._generate_alert(smoothed, roc)

        estimate = GlucoseEstimate(
            timestamp=reading.timestamp,
            glucose_mgdl=round(smoothed, 1),
            confidence=confidence,
            trend=trend,
            alert=alert
        )
        self.history.append(estimate)
        if len(self.history) > 500:
            self.history.pop(0)

        return estimate

    def _generate_alert(self, glucose: float, roc: float) -> Optional[str]:
        if glucose < self.HYPO_THRESHOLD:
            return f"⚠ HYPOGLYCEMIA: {glucose:.0f} mg/dL — consume fast-acting glucose immediately!"
        if glucose > self.HYPER_THRESHOLD:
            return f"⚠ HYPERGLYCEMIA: {glucose:.0f} mg/dL — consider correction and reduce carbs."
        if glucose < self.NORMAL_LOW and roc < -0.5:
            return f"⚡ Glucose dropping ({glucose:.0f} mg/dL, {roc:.1f}/cycle) — potential hypo risk."
        if glucose > 130 and roc > 1.5:
            return f"⚡ Glucose rising fast ({glucose:.0f} mg/dL, +{roc:.1f}/cycle) — monitor closely."
        return None

    def get_summary(self) -> dict:
        """Returns latest state as a dict (used by Flask and voice handler)."""
        if not self.history:
            return {"glucose": None, "trend": "unknown", "alert": None}
        latest = self.history[-1]
        return {
            "glucose":    latest.glucose_mgdl,
            "trend":      latest.trend,
            "confidence": latest.confidence,
            "alert":      latest.alert,
            "timestamp":  latest.timestamp,
            "history":    [e.glucose_mgdl for e in self.history[-60:]]
        }

# ─── Simulation Mode (demo / no hardware) ───────────────────────────────────

class GlucoTwinSimulator:
    """
    Generates realistic synthetic sensor data for demo purposes.
    Use when no hardware is connected.
    """
    def __init__(self):
        self.twin    = GlucoTwin(initial_glucose=95.0)
        self._t      = 0
        self._base_g = 95.0

    def _simulate_meal(self, t: float) -> float:
        """Post-meal glucose spike at t=50, t=200"""
        spike1 = 40 * np.exp(-0.01 * (t - 50)**2) if t > 30 else 0
        spike2 = 30 * np.exp(-0.008 * (t - 200)**2) if t > 180 else 0
        return spike1 + spike2

    def step(self) -> GlucoseEstimate:
        self._t += 1
        t = self._t

        # Simulate PPG values that encode glucose
        true_glucose = self._base_g + self._simulate_meal(t) + np.random.normal(0, 1)
        ir_value  = int(18000 - (true_glucose - 100) / 0.015 + np.random.normal(0, 200))
        red_value = int(ir_value / (0.95 + true_glucose * 0.0002))

        reading = SensorReading(
            timestamp  = time.time(),
            ppg_ir     = max(5000, ir_value),
            ppg_red    = max(5000, red_value),
            heart_rate = 70 + np.random.normal(0, 3),
            spo2       = 98.5 + np.random.normal(0, 0.3),
            accel_x    = np.random.normal(0, 0.1),
            accel_y    = np.random.normal(0, 0.1),
            accel_z    = 9.81 + np.random.normal(0, 0.05),
            skin_temp  = 33.5 + np.random.normal(0, 0.2),
            ecg_value  = 512 + int(50 * np.sin(2 * np.pi * t / 20))
        )
        return self.twin.ingest(reading)


# ─── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Gluco Twin — Digital Twin Simulation\n" + "=" * 40)
    sim = GlucoTwinSimulator()
    for i in range(30):
        est = sim.step()
        bar = "█" * int(est.glucose_mgdl / 10)
        alert_str = f"  ← {est.alert}" if est.alert else ""
        print(f"[{i+1:3d}] {est.glucose_mgdl:6.1f} mg/dL  {est.trend:20s}  conf:{est.confidence:.2f}  {bar}{alert_str}")
        time.sleep(0.1)
