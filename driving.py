import threading
import time
import mycamera
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gpiozero import DigitalOutputDevice
from gpiozero import PWMOutputDevice

PWMA = PWMOutputDevice(18)
AIN1 = DigitalOutputDevice(22)
AIN2 = DigitalOutputDevice(27)

PWMB = PWMOutputDevice(23)
BIN1 = DigitalOutputDevice(25)
BIN2 = DigitalOutputDevice(24)

def motor_go(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_back(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_left(speed):
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = speed
    BIN1.value = 0
    BIN2.value = 1
    PWMB.value = speed

def motor_right(speed):
    AIN1.value = 1
    AIN2.value = 0
    PWMA.value = speed
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = speed

def motor_stop():
    AIN1.value = 0
    AIN2.value = 1
    PWMA.value = 0.0
    BIN1.value = 1
    BIN2.value = 0
    PWMB.value = 0.0

speedSet = 0.8

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              66: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value
        
def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height/2):,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (200,66))
    image = cv2.GaussianBlur(image, (5,5), 0)
    _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

camera = mycamera.MyPiCamera(640, 480)

_, image = camera.read()
image = cv2.flip(image, -1)
imagednn = image
image_ok = 0
image_find_ok = 0

def opencvdnn_thread():
    global image, imagednn
    global image_ok, image_find_ok

    model = cv2.dnn.readNetFromTensorflow('/home/qdbwls/AI_CAR/OpencvDnn/OpencvDnn-master/models/frozen_inference_graph.pb',
                                          '/home/qdbwls/AI_CAR/OpencvDnn/OpencvDnn-master/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    while True:
        if image_ok == 1:
            imagednn = image
            image_height, image_width, _ = imagednn.shape

            model.setInput(cv2.dnn.blobFromImage(imagednn, size=(300, 300), swapRB=True))
            output = model.forward()
            
            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > .5:
                    class_id = detection[1]
                    class_name = id_class_name(class_id, classNames)
                    print(str(str(class_id) + " " + str(detection[2]) + " " + class_name))
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    cv2.rectangle(imagednn, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                    cv2.putText(imagednn, class_name, (int(box_x), int(box_y + 0.05*image_height)), cv2.FONT_HERSHEY_SIMPLEX, (.005*image_width), (0, 0, 255))
            image_find_ok = 1

def main():
    global image, imagednn
    global image_ok, image_find_ok

    model_path = '/home/qdbwls/AI_CAR/model/lane_navigation_final.keras'
    model = load_model(model_path)
    carState = "stop"
    
    try:
        while True:
            keValue = cv2.waitKey(1)
            
            if keValue == ord('q'):
                break
            elif keValue == 82:
                print("go")
                carState = "go"
            elif keValue == 84:
                print("stop")
                carState = "stop"

            image_ok = 0
            _, image = camera.read()
            image = cv2.flip(image, -1)
            image_ok = 1

            preprocessed = img_preprocess(image)
            cv2.imshow('pre', preprocessed)
            
            if image_find_ok == 1:
                cv2.imshow('imagednn',imagednn)
                image_find_ok = 0

            X = np.asarray([preprocessed])
            steering_angle = int(model.predict(X)[0])
            print('predict angle:', steering_angle)

            if carState == "go":
                if steering_angle >= 75 and steering_angle <= 110:
                    print("go")
                    motor_go(speedSet)
                elif steering_angle > 111:
                    print('right')
                    motor_right(speedSet)
                elif steering_angle < 76:
                    print('left')
                    motor_left(speedSet)
            elif carState == "stop":
                motor_stop()
    except Keyboardinterrupt:
        pass

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    task1 = threading.Thread(target=opencvdnn_thread)
    task1.start()
    
    main()
    
    cv2.destroyAllWindows()