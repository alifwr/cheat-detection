import mtcnn
import cv2
import math
import time
from Alarm import Alarm

def millis():
    return round(time.time() * 1000)

# cam = cv2.VideoCapture('sample.mp4')
cam = cv2.VideoCapture(0)
detector = mtcnn.MTCNN()
alarm = Alarm()

frame_width = int(cam.get(3))
frame_height = int(cam.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 50, (frame_width,frame_height))
last_time = 0
cheating = False

while(True):
    _, img = cam.read()
    img = cv2.flip(img, 1)

    faces = detector.detect_faces(img)

    for face in faces:
        x, y, w, h = face['box']
        keypoints = face['keypoints']
        cv2.line(img, keypoints['left_eye'],
                 keypoints['right_eye'], (255, 0, 0), 1)
        # cv2.line(img, keypoints['mouth_left'], keypoints['mouth_right'], (0,0,255), 1)
        cv2.circle(img, keypoints['nose'], 2, (255, 255, 0), 1)

        l_dis = math.sqrt(pow(keypoints['left_eye'][0] - keypoints['nose'][0], 2) + pow(
            keypoints['left_eye'][1] - keypoints['nose'][1], 2))
        r_dis = math.sqrt(pow(keypoints['right_eye'][0] - keypoints['nose'][0], 2) + pow(
            keypoints['right_eye'][1] - keypoints['nose'][1], 2))

        if(abs(l_dis-r_dis) < max(l_dis, r_dis)/3):
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cheating = False
            last_time = millis()
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cheating = True
        # print(keypoints)

        # out.write(img)

    if(cheating and millis()-last_time>5000):
        print("CHEATING")
        alarm.start()
    else:
        print("NOT CHEAT")
        alarm.stop()

    cv2.imshow('img', img)

    key = cv2.waitKey(1)

    if(key == ord('q')):
        break
