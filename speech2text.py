import speech_recognition as sr


r = sr.Recognizer()
m = sr.Microphone(chunk_size=8192)

try:
    print("A moment of silence, please...")
    with m as source: r.adjust_for_ambient_noise(source)
    print("Set minimum energy threshold to {}".format(r.energy_threshold))
    while True:
        print("Say something!")
        try:
            with m as source: audio = r.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("Timed out, listening again...")
            continue
        print("Got it! Now to recognize it...")
        try:
            value = r.recognize_vosk(audio)
            print("You said {}".format(value))
        except:
            print("Oops! Didn't catch that")
except KeyboardInterrupt:
    pass
