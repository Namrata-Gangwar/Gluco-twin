"""
Gluco Twin — Main Entry Point
Starts sensor manager, voice assistant, and Flask dashboard together.

Usage:
    python main.py             # full system (dashboard + voice)
    python main.py --sim       # force simulation mode
    python main.py --no-voice  # disable voice (useful for testing)
    python main.py --port /dev/ttyACM0  # specify Arduino port
"""

import argparse
import threading
import time
import sys

def main():
    parser = argparse.ArgumentParser(description="Gluco Twin Digital Twin")
    parser.add_argument("--port",     default="/dev/ttyUSB0", help="Arduino serial port")
    parser.add_argument("--sim",      action="store_true",    help="Force simulation mode")
    parser.add_argument("--no-voice", action="store_true",    help="Disable voice assistant")
    parser.add_argument("--vosk-model", default="vosk-model-small-en-in-0.4",
                        help="Path to Vosk model directory")
    args = parser.parse_args()

    print("=" * 50)
    print("  GLUCO TWIN — Digital Twin Glucose Monitor")
    print("=" * 50)

    # ── Start sensor manager ────────────────────────────────────────────────
    from sensor_reader import SensorManager
    port = "/dev/null" if args.sim else args.port
    manager = SensorManager(port=port)
    manager.start()
    print(f"[MAIN] Sensor mode: {manager.mode}")

    # Wait for first reading
    print("[MAIN] Waiting for first sensor reading...")
    for _ in range(20):
        data = manager.get_latest()
        if data and data.get("glucose"):
            print(f"[MAIN] First glucose estimate: {data['glucose']:.1f} mg/dL")
            break
        time.sleep(0.5)

    # ── Start Flask dashboard in background ─────────────────────────────────
    import dashboard
    dashboard.manager = manager

    def run_flask():
        dashboard.app.run(host="0.0.0.0", port=5000, debug=False,
                          threaded=True, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[MAIN] Dashboard started at http://localhost:5000")

    # ── Start voice assistant ────────────────────────────────────────────────
    if not args.no_voice:
        from voice_handler import GlucoVoiceAssistant
        assistant = GlucoVoiceAssistant(
            twin=manager.twin,
            vosk_model_path=args.vosk_model
        )

        def run_voice():
            assistant.run()

        voice_thread = threading.Thread(target=run_voice, daemon=True)
        voice_thread.start()
        print("[MAIN] Voice assistant started. Say 'glucose' to activate.")
    else:
        print("[MAIN] Voice assistant disabled.")

    print("\n[MAIN] System running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(5)
            data = manager.get_latest()
            if data and data.get("glucose"):
                alert = f" | {data['alert']}" if data.get("alert") else ""
                print(f"[LIVE] {data['glucose']:.1f} mg/dL  {data['trend']}{alert}")
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down Gluco Twin. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
