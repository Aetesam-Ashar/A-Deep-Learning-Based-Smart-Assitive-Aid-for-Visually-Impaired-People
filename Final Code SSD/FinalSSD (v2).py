import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from datetime import datetime
import pytesseract
from PIL import Image
import subprocess
import os

# GPIO setup for ultrasonic sensor
TRIG_PIN = 23
ECHO_PIN = 24
# Define GPIO pins for LED and Buzzer
LED_PIN = 18
BUZZER_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Directory for saving captured images
IMAGE_DIR = "/home/pi/Desktop/Raspberry Pi Test/Captured Images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load class names from COCO dataset
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco2.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load the SSD MobileNet V3 model
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def measure_distance():
    """Measures distance from the ultrasonic sensor."""
    GPIO.output(TRIG_PIN, False)
    time.sleep(0.2)
    GPIO.output(TRIG_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, False)
    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()
    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = round(pulse_duration * 17150, 2)
    return distance

def detect_objects(frame, distance):
    """Performs object detection on the given frame and overlays detection information including distance."""
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    detected_objects = []
    detection_texts = []

    if len(classIds) != 0:
        # Flatten arrays to ensure we handle them as simple list-like structures
        classIds = classIds.flatten()
        confs = confs.flatten()
        for classId, confidence, box in zip(classIds, confs, bbox):
            if confidence > 0.5:
                # Ensure classId is an integer using int()
                if len(detected_objects) < 3:  # Limit detections to 3
                    object_label = classNames[int(classId) - 1]
                    detection_text = f"{object_label}"
                    detected_objects.append((object_label, detection_text))
                    detection_texts.append(detection_text)  # Collect text for combined output
                    cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{object_label}: {confidence*100:.2f}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    break

    combined_text = ", ".join(detection_texts) + f" at {distance:.1f}cm" if detection_texts else ""
    return combined_text, frame

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_path = "/home/pi/Desktop/Raspberry Pi Test/Captured Images/preprocessed_" + os.path.basename(image_path)
    cv2.imwrite(preprocessed_path, thresh)
    return preprocessed_path


camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))

try:
    while True:
        distance = measure_distance()
        print(f"Measured distance: {distance} cm")

        if 5 < distance < 100:  # Trigger capture within specified distance range
            GPIO.output(LED_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            
            camera.capture(rawCapture, format="bgr")
            image = rawCapture.array
            detection_text, detected_image = detect_objects(image, distance)
            
            GPIO.output(LED_PIN, GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.LOW)

            if detection_text:
                print(f"Detected objects: {detection_text}")
                subprocess.call(['python3', 'speech.py', detection_text])
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(IMAGE_DIR, f"detected_{timestamp}.jpg")
            cv2.imwrite(save_path, detected_image)

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
            
            cv2.imshow("Detected Objects", detected_image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

            rawCapture.truncate(0)

        time.sleep(3)

finally:
    GPIO.cleanup()