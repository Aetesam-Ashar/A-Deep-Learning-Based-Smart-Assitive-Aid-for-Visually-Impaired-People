
import cv2
import numpy as np
import pygame
pygame.mixer.init()
# to speech conversion
# LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3

from gtts import gTTS 
import os 
import RPi.GPIO as GPIO
import time
import pytesseract
from PIL import Image
import time

import time

GPIO.setmode(GPIO.BCM)
 
GPIO_TRIGGER = 23
GPIO_ECHO = 24
LED = 18
Buzzer = 17
GPIO.setup(LED,GPIO.OUT)
GPIO.output(LED,GPIO.LOW)

GPIO.setup(Buzzer,GPIO.OUT)
GPIO.output(Buzzer,GPIO.LOW)

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
 
def distance():
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance
 

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1 ] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))




# Check if the webcam is opened correctly

while True:
    dist = distance()
    print ("Measured Distance = %.1f cm" % dist)
    time.sleep(1)
    GPIO.output(LED,GPIO.LOW)
    GPIO.output(Buzzer,GPIO.LOW)
    if dist>10 and dist<150:
        
        GPIO.output(LED, GPIO.HIGH)
        GPIO.output(Buzzer, GPIO.HIGH)
        time.sleep(1)  # Keep the buzzer on for 1 second
        GPIO.output(Buzzer, GPIO.LOW)  # Then turn off the buzzer
        cap = cv2.VideoCapture(0)
        ret, img = cap.read()
        cv2.imwrite('3.png',img)
        cap.release()
        #camera.capture('1.jpg')
        #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        #cv2.imshow('Input', frame)
        # Loading image
        #img = cv2.imread("1.jpg")
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
                # Showing information on the screen
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.55:
                   # Object detected
                   center_x = int(detection[0] * width)
                   center_y = int(detection[1] * height)
                   w = int(detection[2] * width)
                   h = int(detection[3] * height)

                   # Rectangle coordinates
                   x = int(center_x - w / 2)
                   y = int(center_y - h / 2)

                   boxes.append([x, y, w, h])
                   confidences.append(float(confidence))
                   class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if indexes is not None and len(indexes) > 0:
            indexes = indexes.flatten()  # Ensuring it is flattened properly
            for i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                confidence_percent = confidences[i] * 100  # Convert to percentage
                label_with_conf = '{}: {:.2f}%'.format(label, confidence_percent)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label_with_conf, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)
                print(label + ' Detected at ' + str(dist))

                GPIO.output(LED,GPIO.LOW)
                GPIO.output(Buzzer,GPIO.LOW)
                myobj = gTTS(text=label+' Detected at '+str(int(dist))+'cm',lang='en') 
              #myobj.save(f'speech{count%2}.mp3')
              # Saving the converted audio in a mp3 file named 
              # welcome  
                myobj.save('voice.mp3')
                pygame.mixer.music.load("voice.mp3")
                pygame.mixer.music.play()
                os.remove("voice.mp3")
                time.sleep(1.5) #previously 2.25
              #if label=='person':
                  #message = client.messages.create(from_='+19784814504',body =loc,to ='+918892209021')
              #mixer.init()
              #mixer.music.load('voice.mp3')
               #mixer.music.play()
              #mixer.remove('voice.mp3')



        #cv2.imshow("Image", img)
        #cv2.imwrite("Image.jpg", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


        c = cv2.waitKey(1)
        #if c == 10:
            #break
        
        GPIO.output(LED,GPIO.LOW)
        img =Image.open ('3.png')
        text = pytesseract.image_to_string(img, config='')
        print (text)
        myobj = gTTS(text=' Detected text is '+text,lang='en')  
        myobj.save('voice.mp3')
        pygame.mixer.music.load("voice.mp3")
        pygame.mixer.music.play()
        os.remove("voice.mp3")
        if len(text)>20:
            time.sleep(2)
        else:
            time.sleep(2)
    else:
        print('No Object detected');
        #myobj = gTTS(text='No Object detected',lang='en') 
        #myobj.save(f'speech{count%2}.mp3')
        # Saving the converted audio in a mp3 file named 
        # welcome  
        #myobj.save('voice.mp3')
        #pygame.mixer.music.load("voice.mp3")
        #pygame.mixer.music.play()
        #os.remove("voice.mp3")
        time.sleep(1)


cv2.destroyAllWindows()
