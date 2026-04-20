import speech_recognition as sr
import pyttsx3
import webbrowser
import os


engine = pyttsx3.init()

def speak(text):
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio).lower()
        print("You said:", command)
        return command
    except:
        return ""


def run_command(command):
    if "open youtube" in command:
        speak("Opening YouTube")
        webbrowser.open("https://www.youtube.com")

    elif "open google" in command:
        speak("Opening Google")
        webbrowser.open("https://www.google.com")

    elif "open github" in command:
        speak("Opening GitHub")
        webbrowser.open("https://github.com")

    elif "open code" in command:
        speak("Opening VS Code")
        os.system("code")

    elif "time" in command:
        from datetime import datetime
        now = datetime.now().strftime("%H:%M")
        speak(f"Current time is {now}")

    elif "exit" in command:
        speak("Goodbye")
        exit()

    else:
        speak("I did not understand")



speak("Voice control started")

while True:
    command = listen()
    if command:
        run_command(command)