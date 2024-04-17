import RPi.GPIO as GPIO  # Imports the GPIO library to control GPIO pins of Raspberry Pi
import time  # Imports the time library for sleep/delay functions
import cv2  # Imports OpenCV for image processing
import numpy as np  # Imports NumPy for numerical operations
from picamera.array import PiRGBArray  # Allows capture of images from PiCamera in NumPy arrays
from picamera import PiCamera  # Imports the PiCamera library
from datetime import datetime  # Imports datetime for handling date and time operations
import pytesseract  # Imports the Tesseract-OCR Python library for OCR
from PIL import Image  # Imports the Python Imaging Library for image handling
import subprocess  # Allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes
import os  # Provides a way of using operating system dependent functionality

# GPIO setup for ultrasonic sensor
TRIG_PIN = 23  # GPIO pin for the ultrasonic sensor's trigger
ECHO_PIN = 24  # GPIO pin for the ultrasonic sensor's echo
LED_PIN = 18  # GPIO pin for the LED
BUZZER_PIN = 17  # GPIO pin for the buzzer
GPIO.setmode(GPIO.BCM)  # Sets the GPIO pin numbering system to BCM
GPIO.setup(TRIG_PIN, GPIO.OUT)  # Sets the trigger pin as an output
GPIO.setup(ECHO_PIN, GPIO.IN)  # Sets the echo pin as an input
GPIO.setup(LED_PIN, GPIO.OUT)  # Sets the LED pin as an output
GPIO.setup(BUZZER_PIN, GPIO.OUT)  # Sets the buzzer pin as an output

# Directory for saving captured images
IMAGE_DIR = "/home/pi/Desktop/Raspberry Pi Test/Captured Images"  # Path to the directory where images will be saved
os.makedirs(IMAGE_DIR, exist_ok=True)  # Creates the directory if it doesn't exist

# Load class names from COCO dataset
classNames = []  # Initializes an empty list to store class names
classFile = "/home/pi/Desktop/Object_Detection_Files/coco2.names"  # Path to the file containing class names
with open(classFile, "rt") as f:  # Opens the class file in read-text mode
    classNames = f.read().rstrip("\n").split("\n")  # Reads and splits the file into a list of class names

# Load the SSD MobileNet V3 model
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"  # Path to the model's configuration file
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"  # Path to the model's trained weights
net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Loads the model from files
net.setInputSize(320, 320)  # Sets the input size for the model
net.setInputScale(1.0 / 127.5)  # Sets the input scaling
net.setInputMean((127.5, 127.5, 127.5))  # Sets the mean for input normalization
net.setInputSwapRB(True)  # Sets the model to swap Red and Blue channels

def measure_distance():
    """Measures distance from the ultrasonic sensor."""
    GPIO.output(TRIG_PIN, False)  # Ensures the trigger pin is low
    time.sleep(0.2)  # Delays for sensor settling
    GPIO.output(TRIG_PIN, True)  # Sets the trigger pin high
    time.sleep(0.00001)  # Trigger pulse of 10 microseconds
    GPIO.output(TRIG_PIN, False)  # Sets the trigger pin low
    while GPIO.input(ECHO_PIN) == 0:  # Waits for the echo pin to go high
        pulse_start = time.time()  # Records the last "low" time
    while GPIO.input(ECHO_PIN) == 1:  # Waits for the echo pin to go low
        pulse_end = time.time()  # Records the last "high" time
    pulse_duration = pulse_end - pulse_start  # Duration of the echo pulse
    distance = round(pulse_duration * 17150, 2)  # Converts pulse duration to distance
    return distance  # Returns the calculated distance

def detect_objects(frame):
    """Performs object detection on the given frame."""
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)  # Detects objects in the frame
    detected_objects = []  # Initializes a list to hold detected objects
    if len(classIds) != 0:  # Checks if any objects were detected
        for classId, confidence, box in zip(classIds, confs, bbox):  # Iterates over detected objects
            if confidence[0] > 0.5:  # Filters out detections with low confidence
                object_label = classNames[classId[0]-1]  # Gets the object label from classNames list
                detected_objects.append(object_label)  # Adds the object label to the list of detected objects
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)  # Draws a rectangle around the object
                cv2.putText(frame, f"{object_label}: {confidence[0]*100:.2f}%",  # Puts the label and confidence on the frame
                            (box[0]+10, box[1]+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return detected_objects, frame  # Returns the list of detected objects and the modified frame

def preprocess_image(image_path):
    """Preprocesses an image for OCR."""
    img = cv2.imread(image_path)  # Reads the image from disk
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Converts the image to grayscale
    # Applying Otsu's thresholding to binarize the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_path = "/home/pi/Desktop/Raspberry Pi Test/Captured Images/preprocessed_" + os.path.basename(image_path)  # Sets path for saving preprocessed image
    cv2.imwrite(preprocessed_path, thresh)  # Writes the preprocessed image to disk
    return preprocessed_path  # Returns the path of the preprocessed image

camera = PiCamera()  # Initializes the camera
camera.resolution = (640, 480)  # Sets the resolution of the camera
rawCapture = PiRGBArray(camera, size=(640, 480))  # Initializes a PiRGBArray to hold the image

try:
    while True:  # Main loop
        distance = measure_distance()  # Calls the distance measurement function
        print(f"Measured distance: {distance} cm")  # Prints the measured distance

        if 5 < distance < 250:  # Checks if the distance is within the specified range
            GPIO.output(LED_PIN, GPIO.HIGH)  # Turns on the LED
            GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turns on the buzzer
            
            camera.capture(rawCapture, format="bgr")  # Captures an image in BGR format
            image = rawCapture.array  # Converts the captured image to a NumPy array
            detected_objects, detected_image = detect_objects(image)  # Calls the object detection function
            
            GPIO.output(LED_PIN, GPIO.LOW)  # Turns off the LED
            GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turns off the buzzer

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Gets the current timestamp
            save_path = os.path.join(IMAGE_DIR, f"detected_{timestamp}.jpg")  # Sets the path for saving the detected image
            cv2.imwrite(save_path, detected_image)  # Saves the detected image
            
            if detected_objects:  # Checks if any objects were detected
                print(f"Detected objects: {', '.join(detected_objects)}")  # Prints the detected objects
                for obj in set(detected_objects):  # Iterates over unique detected objects
                    subprocess.call(['python3', 'speech.py', obj])  # Calls an external script to convert text to speech
            
            # Now, perform OCR on every captured image
            print("Detecting text...")
            preprocessed_image_path = preprocess_image(save_path)  # Calls the image preprocessing function
            img_for_ocr = Image.open(preprocessed_image_path)  # Opens the preprocessed image
            recognized_text = pytesseract.image_to_string(img_for_ocr)  # Extracts text from the image using OCR
            if recognized_text.strip():  # Checks if any text was recognized
                print(f"Recognized text: {recognized_text}")  # Prints the recognized text
                subprocess.call(['python3', 'speech.py', recognized_text])  # Converts the recognized text to speech
            else:
                print("No text detected.")  # Indicates no text was detected

            cv2.imshow("Detected Objects", detected_image)  # Displays the detected image
            cv2.waitKey(2000)  # Waits 2000 ms before closing the window
            cv2.destroyAllWindows()  # Destroys all OpenCV windows

            rawCapture.truncate(0)  # Clears the contents of rawCapture to prepare for next capture

        time.sleep(3)  # Delays for 3 seconds before next loop iteration

finally:
    GPIO.cleanup()  # Cleans up the GPIO to ensure all resources are freed