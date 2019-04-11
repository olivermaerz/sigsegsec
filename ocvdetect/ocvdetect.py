import cv2 as cv
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Configure the min probability
min_probability = 0.5

camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(1280, 720))

# 90 labels for the mobilenet v2
# (from https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt)
labels = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
          7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign',
          14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
          21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella',
          31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
          37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
          42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
          49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
          56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair',
          63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop',
          74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
          80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
          88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

cvNet = cv.dnn.readNetFromTensorflow('models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                                     'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

# Code for cv2 based on https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
# Weights and config: MobileNet-SSD v2  Version 2018_03_29

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    start_time = time.time()  # start time of the loop
    image = frame.array
    rows, cols, _ = image.shape
    cvNet.setInput(cv.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()
    # clear the stream
    rawCapture.truncate(0)

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        if score > min_probability:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (100, 100, 210), thickness=1)
            label = labels[detection[1]]
            cv.putText(image, label, (int(left), abs(int(top)-5)), cv.FONT_HERSHEY_PLAIN, 1, (10, 10, 255))

    # print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
    cv.imshow('Live Image', image)
    # cv.imwrite("image.jpg", image)
    if cv.waitKey(1) == ord('q'):
        break
