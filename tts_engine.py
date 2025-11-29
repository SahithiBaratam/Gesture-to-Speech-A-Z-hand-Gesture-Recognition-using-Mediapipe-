import pyttsx3
import threading

def speak(text):
    if not text or not text.strip():
        return

    def run():
        engine = pyttsx3.init()   # NEW engine each time → fixes non-speaking issue
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run, daemon=True).start()
