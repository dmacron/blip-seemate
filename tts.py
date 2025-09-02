import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    sample_caption = "A cat sitting on a sofa."
    speak(sample_caption)