import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
import webbrowser
import os

# Model yuklash
model = Model("model/vosk-model-small-uz-0.22")

q = queue.Queue()

def callback(indata, frames, time, status):
    q.put(bytes(indata))


# Buyruqlar
def run_command(text):
    print("You said:", text)

    if "yutib ni och" in text:
        print("Opening YouTube")
        webbrowser.open("https://youtube.com")

    elif "gogolni och" in text:
        print("Opening Google")
        webbrowser.open("https://google.com")

    elif "code" in text:
        print("Opening VS Code")
        os.system("code")

    elif "tizimdan chiq" in text:
        print("Exiting...")
        exit()


# Recognizer
recognizer = KaldiRecognizer(model, 16000)

# Microphone stream
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):

    print("🎤 Speak something...")

    while True:
        data = q.get()

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")

            if text:
                run_command(text)