"""
Gluco Twin — Flask Dashboard
Real-time glucose monitoring web UI for judges.

Run:  python dashboard.py
Open: http://localhost:5000  (or http://<Pi-IP>:5000 from another device)
"""

from flask import Flask, render_template_string, jsonify
from sensor_reader import SensorManager
import threading, time

app = Flask(__name__)
manager = SensorManager(port="/dev/ttyUSB0")

# ─── HTML Dashboard ──────────────────────────────────────────────────────────

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gluco Twin — Digital Twin Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --yellow: #d29922; --red: #f85149; --blue: #58a6ff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', system-ui, sans-serif; padding: 1.5rem; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.25rem; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 10px; padding: 1rem 1.25rem; }
  .card .label { font-size: 0.75rem; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  .card .value { font-size: 2rem; font-weight: 700; line-height: 1; }
  .card .unit  { font-size: 0.85rem; color: var(--muted); margin-left: 2px; }
  .card .status { font-size: 0.8rem; margin-top: 6px; }
  .normal  { color: var(--green); }
  .warning { color: var(--yellow); }
  .danger  { color: var(--red); }
  .chart-card { background: var(--card); border: 1px solid var(--border);
                border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem; }
  .chart-card h2 { font-size: 0.9rem; color: var(--muted); margin-bottom: 1rem; font-weight: 500; }
  canvas { max-height: 220px; }
  .alert-box { border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.9rem;
               margin-bottom: 1.5rem; display: none; }
  .alert-box.danger  { background: #2d1117; border: 1px solid var(--red); color: var(--red); }
  .alert-box.warning { background: #2d1f00; border: 1px solid var(--yellow); color: var(--yellow); }
  .mode-badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
                font-size: 0.75rem; font-weight: 600; margin-left: 10px;
                vertical-align: middle; }
  .mode-hw  { background: #1a3a1a; color: var(--green); border: 1px solid var(--green); }
  .mode-sim { background: #1a2a3a; color: var(--blue); border: 1px solid var(--blue); }
  .voice-panel { background: var(--card); border: 1px solid var(--border);
                 border-radius: 10px; padding: 1.25rem; }
  .voice-panel h2 { font-size: 0.9rem; color: var(--muted); margin-bottom: 0.75rem; font-weight: 500; }
  .voice-log { font-family: monospace; font-size: 0.8rem; color: var(--muted);
               max-height: 120px; overflow-y: auto; }
  .voice-log .vl-entry { margin-bottom: 2px; }
  .voice-log .vl-q { color: var(--blue); }
  .voice-log .vl-a { color: var(--green); }
  footer { margin-top: 2rem; font-size: 0.75rem; color: var(--muted); text-align: center; }
</style>
</head>
<body>
<h1>Gluco Twin <span id="mode-badge" class="mode-badge mode-sim">SIM</span></h1>
<div class="subtitle">Non-invasive digital twin glucose monitor · Real-time dashboard</div>

<div id="alert-box" class="alert-box danger"></div>

<div class="grid">
  <div class="card" id="card-glucose">
    <div class="label">Predicted Glucose</div>
    <div><span class="value" id="glucose">--</span><span class="unit">mg/dL</span></div>
    <div class="status" id="glucose-status">Waiting...</div>
  </div>
  <div class="card">
    <div class="label">Trend</div>
    <div class="value" id="trend" style="font-size:1.3rem;margin-top:4px">--</div>
    <div class="status normal" id="roc">-- mg/dL/min</div>
  </div>
  <div class="card">
    <div class="label">Heart Rate</div>
    <div><span class="value" id="hr">--</span><span class="unit">bpm</span></div>
    <div class="status normal">From MAX30102</div>
  </div>
  <div class="card">
    <div class="label">SpO₂</div>
    <div><span class="value" id="spo2">--</span><span class="unit">%</span></div>
    <div class="status normal">Pulse oximetry</div>
  </div>
  <div class="card">
    <div class="label">Skin Temp</div>
    <div><span class="value" id="temp">--</span><span class="unit">°C</span></div>
    <div class="status normal">Signal compensation</div>
  </div>
  <div class="card">
    <div class="label">Confidence</div>
    <div><span class="value" id="conf">--</span><span class="unit">%</span></div>
    <div class="status normal">Model certainty</div>
  </div>
</div>

<div class="chart-card">
  <h2>Glucose Trend (last 60 readings)</h2>
  <canvas id="glucoseChart"></canvas>
</div>

<div class="voice-panel">
  <h2>Voice Interaction Log</h2>
  <div class="voice-log" id="voice-log">
    <div class="vl-entry" style="color:#555">Waiting for voice commands...</div>
  </div>
</div>

<footer>Gluco Twin v1.0 · Raspberry Pi 4 · Kalman Filter Digital Twin · Vosk Offline STT</footer>

<script>
const ctx = document.getElementById('glucoseChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      label: 'Glucose (mg/dL)',
      data: [],
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.08)',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.4,
      fill: true,
    }, {
      label: 'Hypo threshold (70)',
      data: [],
      borderColor: '#f85149',
      borderWidth: 1,
      borderDash: [6,4],
      pointRadius: 0,
      fill: false,
    }, {
      label: 'Hyper threshold (180)',
      data: [],
      borderColor: '#d29922',
      borderWidth: 1,
      borderDash: [6,4],
      pointRadius: 0,
      fill: false,
    }]
  },
  options: {
    responsive: true,
    animation: false,
    plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
    scales: {
      x: { display: false },
      y: {
        min: 50, max: 250,
        grid: { color: '#21262d' },
        ticks: { color: '#8b949e', font: { size: 11 } }
      }
    }
  }
});

let counter = 0;
function poll() {
  fetch('/api/status').then(r => r.json()).then(d => {
    if (!d.glucose) return;

    document.getElementById('glucose').textContent = d.glucose.toFixed(0);
    document.getElementById('trend').textContent   = d.trend || '--';
    document.getElementById('hr').textContent      = (d.hr || '--').toString();
    document.getElementById('spo2').textContent    = (d.spo2 || '--').toString();
    document.getElementById('temp').textContent    = d.temp ? d.temp.toFixed(1) : '--';
    document.getElementById('conf').textContent    = d.confidence ? Math.round(d.confidence * 100) : '--';

    const gs = document.getElementById('glucose-status');
    const gc = document.getElementById('card-glucose');
    if (d.glucose < 70) {
      gs.textContent = '⚠ Hypoglycemia'; gs.className = 'status danger';
    } else if (d.glucose > 180) {
      gs.textContent = '⚠ Hyperglycemia'; gs.className = 'status danger';
    } else if (d.glucose > 140) {
      gs.textContent = '● Slightly elevated'; gs.className = 'status warning';
    } else {
      gs.textContent = '● Normal range'; gs.className = 'status normal';
    }

    const alertBox = document.getElementById('alert-box');
    if (d.alert) {
      alertBox.style.display = 'block';
      alertBox.textContent = d.alert;
      alertBox.className = d.glucose < 70 ? 'alert-box danger' : 'alert-box warning';
    } else {
      alertBox.style.display = 'none';
    }

    const badge = document.getElementById('mode-badge');
    badge.textContent = d.mode === 'hardware' ? 'HARDWARE' : 'SIM';
    badge.className   = 'mode-badge ' + (d.mode === 'hardware' ? 'mode-hw' : 'mode-sim');

    // Chart update
    counter++;
    const h = d.history || [d.glucose];
    chart.data.labels    = h.map((_, i) => i);
    chart.data.datasets[0].data = h;
    chart.data.datasets[1].data = h.map(() => 70);
    chart.data.datasets[2].data = h.map(() => 180);
    chart.update('none');
  }).catch(() => {});
}

function pollVoice() {
  fetch('/api/voice_log').then(r => r.json()).then(d => {
    if (!d.log || !d.log.length) return;
    const el = document.getElementById('voice-log');
    el.innerHTML = d.log.slice(-8).map(e =>
      `<div class="vl-entry"><span style="color:#555">[${e.time}]</span> ` +
      `<span class="vl-q">Q: ${e.query}</span><br>` +
      `<span class="vl-a">A: ${e.response}</span></div>`
    ).join('');
    el.scrollTop = el.scrollHeight;
  }).catch(() => {});
}

setInterval(poll, 1000);
setInterval(pollVoice, 2000);
poll();
</script>
</body>
</html>
"""

# ─── Voice log store ─────────────────────────────────────────────────────────

voice_log = []

def add_voice_log(query: str, response: str):
    voice_log.append({
        "time": time.strftime("%H:%M:%S"),
        "query": query,
        "response": response
    })
    if len(voice_log) > 50:
        voice_log.pop(0)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route("/api/status")
def api_status():
    data = manager.get_latest() or {}
    raw  = manager.raw_sensor
    data["mode"] = manager.mode
    data["hr"]   = raw.get("hr", 72)
    data["spo2"] = raw.get("spo2", 98.2)
    data["temp"] = raw.get("temp", 33.5)
    return jsonify(data)

@app.route("/api/voice_log")
def api_voice_log():
    return jsonify({"log": voice_log})

@app.route("/api/voice_query/<path:query>")
def api_voice_query(query: str):
    """Called by voice_handler to log Q&A to the dashboard."""
    from voice_handler import GlucoVoiceAssistant
    assistant = GlucoVoiceAssistant.__new__(GlucoVoiceAssistant)
    from voice_handler import IntentParser, ResponseGenerator
    assistant.parser    = IntentParser()
    assistant.responder = ResponseGenerator()
    assistant.twin      = manager.twin
    response = assistant.process_query(query)
    add_voice_log(query, response)
    return jsonify({"response": response})

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Gluco Twin Dashboard...")
    manager.start()
    print("Open http://localhost:5000 in your browser.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
