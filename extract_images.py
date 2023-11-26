# Reads a video in data/videos/ and saves the frames in data/images/

import cv2
import numpy as np


VIDEO_PATH = 'data/videos/test2.mp4'

cap = cv2.VideoCapture('data/videos/test2.mp4')
i = 0
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite('data/test_images2/frame' + str(i) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
