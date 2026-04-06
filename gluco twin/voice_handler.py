"""
Gluco Twin — Voice Interface
Offline speech recognition (Vosk) + intent parser + TTS (pyttsx3)

Install:
    pip install vosk pyttsx3 sounddevice numpy
    # Download Vosk model: https://alphacephei.com/vosk/models
    # Use vosk-model-small-en-in-0.4 (Indian English, ~40MB)
"""

import json
import queue
import threading
import time
import re
from typing import Optional, Callable

# ─── TTS Engine ──────────────────────────────────────────────────────────────

class VoiceOutput:
    """Text-to-speech using pyttsx3 (offline, works on Pi)."""

    def __init__(self, rate: int = 160, volume: float = 0.95):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            # Try to use a clearer voice if available
            voices = self.engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'en' in v.id.lower():
                    self.engine.setProperty('voice', v.id)
                    break
            self._available = True
        except Exception as e:
            print(f"[TTS] pyttsx3 not available: {e}. Using print fallback.")
            self._available = False

    def speak(self, text: str):
        print(f"[VOICE OUT] {text}")
        if self._available:
            self.engine.say(text)
            self.engine.runAndWait()

# ─── Intent Parser ───────────────────────────────────────────────────────────

class IntentParser:
    """
    Rule-based NLP intent parser.
    Maps spoken phrases → intent + optional parameters.
    """

    INTENTS = {
        "GET_GLUCOSE": [
            r"what is my glucose",
            r"what('s| is) my (blood )?sugar",
            r"(current|latest|now)? glucose (level|reading)?",
            r"check (my )?glucose",
            r"how (is|are) my (blood )?sugar",
            r"glucose (level|reading|check)",
            r"sugar level",
        ],
        "GET_TREND": [
            r"(what is|what's) my (glucose )?trend",
            r"is my (glucose|sugar) (going up|going down|rising|falling|stable)",
            r"how is (my )?glucose (trending|changing)",
            r"trend",
        ],
        "GET_ALERT_STATUS": [
            r"(am i|are things) (ok|okay|normal|fine|safe)",
            r"any (alerts|warnings|issues)",
            r"(should i|do i need to) worry",
            r"is (my glucose|everything) normal",
            r"status",
        ],
        "GET_HISTORY": [
            r"what was my (last|previous|recent) (reading|glucose|level)",
            r"(last|previous) (few |)?readings",
            r"glucose history",
            r"how has (it been|my glucose been)",
        ],
        "GET_ADVICE": [
            r"what should i (do|eat|drink)",
            r"(any )?advice",
            r"recommendation",
            r"(how to|how can i) (lower|raise|improve) (my )?(glucose|sugar)",
        ],
        "HELP": [
            r"help",
            r"what can (you|i) (do|say|ask)",
            r"commands",
        ],
    }

    def __init__(self):
        self._patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENTS.items()
        }

    def parse(self, text: str) -> Optional[str]:
        text = text.strip().lower()
        for intent, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return intent
        return None

# ─── Response Generator ──────────────────────────────────────────────────────

class ResponseGenerator:
    """Turns a glucose summary dict into a natural spoken response."""

    def generate(self, intent: str, summary: dict) -> str:
        glucose   = summary.get("glucose")
        trend     = summary.get("trend", "unknown")
        alert     = summary.get("alert")
        confidence = summary.get("confidence", 0.8)

        confidence_phrase = ""
        if confidence < 0.6:
            confidence_phrase = "Signal quality is low, so take this with caution. "

        if intent == "GET_GLUCOSE":
            if glucose is None:
                return "I don't have a glucose reading yet. Please make sure the sensor is on your finger."
            response = f"{confidence_phrase}Your current predicted glucose level is {glucose:.0f} milligrams per deciliter."
            if alert:
                response += f" {alert}"
            else:
                response += self._range_comment(glucose)
            return response

        elif intent == "GET_TREND":
            if glucose is None:
                return "Trend data is not available yet. Keep the sensor in place for a few more readings."
            return (f"Your glucose is currently {trend}. "
                    f"The latest reading is {glucose:.0f} milligrams per deciliter.")

        elif intent == "GET_ALERT_STATUS":
            if glucose is None:
                return "I cannot determine your status without a valid sensor reading."
            if alert:
                return f"There is an alert: {alert}"
            return (f"Everything looks normal. Your glucose is {glucose:.0f} milligrams per deciliter, "
                    f"which is within a healthy range, and it is {trend}.")

        elif intent == "GET_HISTORY":
            history = summary.get("history", [])
            if len(history) < 3:
                return "Not enough history yet. I need a few more readings first."
            recent = history[-5:]
            avg    = sum(recent) / len(recent)
            return (f"Your last {len(recent)} readings averaged {avg:.0f} milligrams per deciliter. "
                    f"The most recent was {recent[-1]:.0f}.")

        elif intent == "GET_ADVICE":
            if glucose is None:
                return "Please place your finger on the sensor so I can give personalised advice."
            if glucose < 70:
                return ("Your glucose is critically low. Consume 15 grams of fast-acting carbohydrates "
                        "such as glucose tablets, fruit juice, or regular soda immediately. "
                        "Check again in 15 minutes.")
            elif glucose < 80:
                return ("Your glucose is slightly low. Consider a small snack like crackers or fruit. "
                        "Keep monitoring.")
            elif glucose > 180:
                return ("Your glucose is high. Drink plenty of water, avoid high-carbohydrate foods, "
                        "and consider light physical activity if appropriate for you. "
                        "Consult your doctor if levels stay elevated.")
            elif glucose > 140:
                return ("Glucose is slightly elevated. Reduce simple carbohydrates and "
                        "stay well hydrated. A short walk can help bring it down.")
            else:
                return ("Your glucose is in a healthy range. Maintain your current diet and activity level. "
                        "Stay hydrated and keep up the good work!")

        elif intent == "HELP":
            return ("You can ask me things like: What is my glucose level? "
                    "Is my glucose trending up or down? Am I in the normal range? "
                    "What should I do? Or ask for my recent history.")

        return "I did not understand that. Try asking: What is my glucose level?"

    def _range_comment(self, glucose: float) -> str:
        if glucose < 70:
            return " This is dangerously low. Please act immediately."
        elif glucose < 80:
            return " This is slightly below the normal range."
        elif glucose <= 140:
            return " You are in the healthy range. Keep it up!"
        elif glucose <= 180:
            return " This is slightly elevated. Monitor your diet."
        else:
            return " This is above the normal range. Take action."

# ─── Speech-to-Text (Vosk) ───────────────────────────────────────────────────

class VoiceInput:
    """
    Listens on the microphone and transcribes speech using Vosk (offline).
    Calls on_result(text) when a final result is ready.
    """

    def __init__(self, model_path: str = "vosk-model-small-en-in-0.4",
                 sample_rate: int = 16000):
        self.model_path  = model_path
        self.sample_rate = sample_rate
        self._q          = queue.Queue()
        self._running    = False
        self._thread     = None
        self._available  = self._load_model()

    def _load_model(self) -> bool:
        try:
            from vosk import Model, KaldiRecognizer
            import sounddevice as sd
            self._Model           = Model
            self._KaldiRecognizer = KaldiRecognizer
            self._sd              = sd
            self._model = self._Model(self.model_path)
            print("[STT] Vosk model loaded successfully.")
            return True
        except Exception as e:
            print(f"[STT] Vosk not available: {e}. Using keyboard fallback.")
            return False

    def listen_once(self, timeout: float = 8.0) -> Optional[str]:
        """Block until one spoken phrase is recognised (or timeout). Returns text."""
        if not self._available:
            return self._keyboard_fallback()

        rec = self._KaldiRecognizer(self._model, self.sample_rate)
        result_text = None
        deadline = time.time() + timeout

        def audio_callback(indata, frames, time_info, status):
            self._q.put(bytes(indata))

        with self._sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=4000,
            dtype='int16',
            channels=1,
            callback=audio_callback
        ):
            print("[STT] Listening...")
            while time.time() < deadline:
                try:
                    data = self._q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        result_text = text
                        break

        if result_text:
            print(f"[STT] Recognised: '{result_text}'")
        else:
            print("[STT] No speech detected within timeout.")
        return result_text

    def _keyboard_fallback(self) -> Optional[str]:
        """For testing without a microphone."""
        try:
            text = input("[KEYBOARD FALLBACK] Type your query: ").strip()
            return text if text else None
        except (EOFError, KeyboardInterrupt):
            return None

# ─── Voice Assistant ─────────────────────────────────────────────────────────

class GlucoVoiceAssistant:
    """
    Main voice loop for Gluco Twin.
    Integrates STT → Intent → Response → TTS.

    Usage:
        from digital_twin import GlucoTwin
        twin = GlucoTwin()
        assistant = GlucoVoiceAssistant(twin, wake_word="glucose")
        assistant.run()
    """

    def __init__(self, twin, vosk_model_path: str = "vosk-model-small-en-in-0.4",
                 wake_word: str = "glucose"):
        self.twin      = twin
        self.voice_in  = VoiceInput(model_path=vosk_model_path)
        self.voice_out = VoiceOutput()
        self.parser    = IntentParser()
        self.responder = ResponseGenerator()
        self.wake_word = wake_word.lower()
        self._running  = False

    def process_query(self, text: str) -> str:
        """Full pipeline: text → intent → summary → spoken response."""
        intent = self.parser.parse(text)
        if intent is None:
            return "I did not understand that query. Try asking about your glucose level."
        summary  = self.twin.get_summary()
        response = self.responder.generate(intent, summary)
        return response

    def handle_once(self):
        """Single listen → respond cycle."""
        self.voice_out.speak("Gluco Twin ready. Ask me about your glucose.")
        text = self.voice_in.listen_once(timeout=10)
        if not text:
            self.voice_out.speak("I did not hear anything. Please try again.")
            return
        response = self.process_query(text)
        self.voice_out.speak(response)

    def run(self):
        """
        Continuous loop with wake-word detection.
        Says the wake word aloud to activate → listens for query.
        """
        self._running = True
        print(f"\n[ASSISTANT] Running. Say '{self.wake_word}' to activate.\n")
        self.voice_out.speak(f"Gluco Twin voice assistant started. Say {self.wake_word} to ask a question.")

        while self._running:
            print(f"[ASSISTANT] Waiting for wake word: '{self.wake_word}'...")
            text = self.voice_in.listen_once(timeout=30)
            if text and self.wake_word in text.lower():
                self.voice_out.speak("Yes, how can I help?")
                query = self.voice_in.listen_once(timeout=8)
                if query:
                    response = self.process_query(query)
                    self.voice_out.speak(response)
                else:
                    self.voice_out.speak("I did not catch your question. Please try again.")
            elif text:
                # Check if it's a direct command without wake word
                intent = self.parser.parse(text)
                if intent:
                    response = self.process_query(text)
                    self.voice_out.speak(response)

    def stop(self):
        self._running = False


# ─── CLI Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from digital_twin import GlucoTwinSimulator

    sim       = GlucoTwinSimulator()
    twin      = sim.twin
    assistant = GlucoVoiceAssistant(twin)

    # Warm up with a few readings
    print("Warming up digital twin with 10 readings...")
    for _ in range(10):
        sim.step()

    print("\n─── Testing intent parser ───")
    test_queries = [
        "What is my glucose level?",
        "Is my sugar going up?",
        "Am I okay?",
        "What should I eat?",
        "Show me my recent readings",
        "Help",
    ]
    for q in test_queries:
        response = assistant.process_query(q)
        print(f"\nQ: {q}\nA: {response}")

    print("\n─── Starting interactive voice demo (type queries) ───")
    assistant.handle_once()
