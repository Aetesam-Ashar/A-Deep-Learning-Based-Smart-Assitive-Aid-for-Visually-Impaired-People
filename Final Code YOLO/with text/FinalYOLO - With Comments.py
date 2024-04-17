import cv2  # Imports the OpenCV library for computer vision tasks
import numpy as np  # Imports the NumPy library for numerical operations
import pygame  # Imports the pygame library for playing sounds
pygame.mixer.init()  # Initializes the pygame mixer for playing audio

# to speech conversion
# LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3
# This line is a comment that might refer to a workaround for a bug or dependency issue in certain environments

from gtts import gTTS  # Imports Google Text-to-Speech
import os  # Imports the OS module to interact with the operating system
import RPi.GPIO as GPIO  # Imports the Raspberry Pi GPIO library to control the GPIO pins
import time  # Imports the time module for handling time-related tasks
import pytesseract  # Imports the Python Tesseract library for optical character recognition (OCR)
from PIL import Image  # Imports the Python Imaging Library (PIL) to handle image files

GPIO.setmode(GPIO.BCM)  # Sets the GPIO pin numbering to BCM (Broadcom SOC channel)

GPIO_TRIGGER = 23  # Pin number for the ultrasonic sensor's trigger
GPIO_ECHO = 24  # Pin number for the ultrasonic sensor's echo
LED = 18  # Pin number for the LED
Buzzer = 17  # Pin number for the buzzer
GPIO.setup(LED, GPIO.OUT)  # Sets the LED pin as an output
GPIO.output(LED, GPIO.LOW)  # Sets the LED to low (off)

GPIO.setup(Buzzer, GPIO.OUT)  # Sets the buzzer pin as an output
GPIO.output(Buzzer, GPIO.LOW)  # Sets the buzzer to low (off)

GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # Sets the GPIO trigger as an output
GPIO.setup(GPIO_ECHO, GPIO.IN)  # Sets the GPIO echo as an input
 
def distance():
    """ Measures the distance by using the ultrasonic sensor """
    GPIO.output(GPIO_TRIGGER, True)  # Sends a signal (True means high level)
    time.sleep(0.00001)  # Wait 10 microseconds
    GPIO.output(GPIO_TRIGGER, False)  # Stops sending the signal
 
    StartTime = time.time()  # Records the time at which the signal was sent
    StopTime = time.time()  # Initializes a variable to keep track of when the signal returns
 
    while GPIO.input(GPIO_ECHO) == 0:  # Waits for the echo pin to go high
        StartTime = time.time()
 
    while GPIO.input(GPIO_ECHO) == 1:  # Waits for the echo pin to go low
        StopTime = time.time()
 
    TimeElapsed = StopTime - StartTime  # Calculate the time it took for the echo to return
    distance = (TimeElapsed * 34300) / 2  # Calculate the distance using the speed of sound
 
    return distance  # Returns the measured distance
 
# Load YOLO object detection model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Loads the pre-trained YOLO model
classes = []  # Initializes a list to store the classes
with open("coco.names", "r") as f:  # Opens the file containing the class names
    classes = [line.strip() for line in f.readlines()]  # Reads the class names into a list
layer_names = net.getLayerNames()  # Gets the names of the layers in the network
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]  # Identifies the output layers
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # Generates random colors for the bounding boxes

while True:  # Main program loop
    dist = distance()  # Measure distance
    print("Measured Distance = %.1f cm" % dist)  # Print the measured distance
    time.sleep(1)  # Delay for one second
    GPIO.output(LED, GPIO.LOW)  # Turn off the LED
    GPIO.output(Buzzer, GPIO.LOW)  # Turn off the buzzer
    if dist > 10 and dist < 150:  # If distance is within the range of 10cm to 150cm
        GPIO.output(LED, GPIO.HIGH)  # Turn on the LED
        GPIO.output(Buzzer, GPIO.HIGH)  # Turn on the buzzer
        time.sleep(1)  # Wait for 1 second
        GPIO.output(Buzzer, GPIO.LOW)  # Turn off the buzzer
        cap = cv2.VideoCapture(0)  # Start video capture
        ret, img = cap.read()  # Read an image from the video capture
        cv2.imwrite('3.png', img)  # Save the image to a file
        cap.release()  # Release the video capture object
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # Resize the image
        height, width, channels = img.shape  # Get the dimensions and channels of the image

        # Object detection process
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Create a blob from the image
        net.setInput(blob)  # Set the input for the network
        outs = net.forward(output_layers)  # Get the output from the network

        # Information extraction from the output
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:  # For each output
            for detection in out:  # For each detection
                scores = detection[5:]  # Get the scores
                class_id = np.argmax(scores)  # Get the class ID
                confidence = scores[class_id]  # Get the confidence
                if confidence > 0.55:  # If confidence is greater than 55%
                    # Calculate box dimensions
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Calculate rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])  # Append box coordinates
                    confidences.append(float(confidence))  # Append confidence
                    class_ids.append(class_id)  # Append class ID

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Perform non maximum suppression
        if indexes is not None and len(indexes) > 0:
            indexes = indexes.flatten()  # Flatten the indexes
            for i in indexes:  # For each index
                x, y, w, h = boxes[i]  # Get box coordinates
                label = str(classes[class_ids[i]])  # Get label
                color = colors[i]  # Get color
                confidence_percent = confidences[i] * 100  # Convert confidence to percentage
                label_with_conf = '{}: {:.2f}%'.format(label, confidence_percent)  # Create label with confidence
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Draw rectangle
                cv2.putText(img, label_with_conf, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, color, 3)  # Put text
                print(label + ' Detected at ' + str(dist))  # Print label and distance

                GPIO.output(LED, GPIO.LOW)  # Turn off LED
                GPIO.output(Buzzer, GPIO.LOW)  # Turn off buzzer
                myobj = gTTS(text=label + ' Detected at ' + str(int(dist)) + 'cm', lang='en')  # Create text-to-speech
                myobj.save('voice.mp3')  # Save speech
                pygame.mixer.music.load("voice.mp3")  # Load speech
                pygame.mixer.music.play()  # Play speech
                os.remove("voice.mp3")  # Remove speech file
                time.sleep(1.5)  # Wait for 1.5 seconds

        img = Image.open('3.png')  # Open image file
        text = pytesseract.image_to_string(img, config='')  # Perform OCR
        print(text)  # Print detected text
        myobj = gTTS(text=' Detected text is ' + text, lang='en')  # Create text-to-speech for detected text
        myobj.save('voice.mp3')  # Save speech
        pygame.mixer.music.load("voice.mp3")  # Load speech
        pygame.mixer.music.play()  # Play speech
        os.remove("voice.mp3")  # Remove speech file
        if len(text) > 20:
            time.sleep(2)  # Longer sleep if text is long
        else:
            time.sleep(2)  # Shorter sleep otherwise
    else:
        print('No Object detected')  # Print if no object is detected
        time.sleep(1)  # Wait for 1 second

cv2.destroyAllWindows()  # Destroy all OpenCV windows