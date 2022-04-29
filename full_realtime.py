import mtcnn
import cv2
import math
import time
from Alarm import Alarm
from yolo import YOLO

def millis():
    return round(time.time() * 1000)

cam = cv2.VideoCapture(0)
detector = mtcnn.MTCNN()
yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
yolo.size = int(416)
yolo.confidence = float(0.2)
alarm = Alarm()

x_limit = (35, 75)
y_limit = 65

frame_width = int(cam.get(3))
frame_height = int(cam.get(4))
x_limit = (int((x_limit[0]*frame_width)/100), int((x_limit[1]*frame_width)/100))
y_limit = int((y_limit * frame_height)/100)


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
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
    cv2.rectangle(img, (x_limit[0], y_limit), (x_limit[1], frame_height), (255,255,0), 1)
    width, height, inference_time, results = yolo.inference(img)
    results.sort(key=lambda x: x[2])
    hand_count = 2 #len(results)

    # display hands
    for detection in results[:hand_count]:
        id, name, confidence, x, y, w, h = detection
        print(frame_height, frame_width)
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        if(cx<x_limit[0] or cx>x_limit[1] or cy<y_limit):
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

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
