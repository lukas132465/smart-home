import playsound
import os
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import cv2
from collections import Counter
import time
import vlc
import threading
import speech_recognition as sr


BUFFER_LENGTH = 10
pose_buffer = []
TARGETS = ['Cross', 'Down', 'Other', 'Up']

music_folder = os.path.join(os.path.abspath(""), "music")
file = os.path.join(music_folder, "example.mp3")

instance = vlc.Instance("--no-xlib")
player = instance.media_player_new()
media = instance.media_new(file)
player.set_media(media)
player.audio_set_volume(50)

model = models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

num_classes = 4
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load('mobileNet_fine_tuned.pth'))

cap = cv2.VideoCapture(0)

r = sr.Recognizer()
m = sr.Microphone(chunk_size=8192)

def voice():
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

                if all([word in value for word in ["mark", "play", "pop"]]):
                    print('Starting music')
                    player.play()

            except:
                print("Oops! Didn't catch that")
    except KeyboardInterrupt:
        pass

thread = threading.Thread(target=voice)
thread.start()


while (True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize([0.4093, 0.3851, 0.3808], [0.2469, 0.2507, 0.2426])])
    frame_tensor = transform(frame)
    frame_tensor = frame_tensor.unsqueeze(0)

    output = F.softmax(model(frame_tensor))
    pose_buffer.append(TARGETS[output.argmax(dim=1)])
    if len(pose_buffer) > BUFFER_LENGTH:
        pose_buffer.pop(0)

    counter = Counter(pose_buffer)
    print(counter)
    time.sleep(0.5)

    if counter['Cross'] > 5 and player.is_playing():
        print('Stopping music')
        player.stop()
    elif counter['Up'] > 5 and player.is_playing():
        print('Increasing volume')
        volume = player.audio_get_volume()
        player.audio_set_volume(volume + 10)
    elif counter['Down'] > 5 and player.is_playing():
        print('Decreasing volume')
        volume = player.audio_get_volume()
        player.audio_set_volume(volume + 10)
