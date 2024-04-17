# A-Deep-Learning-Based-Smart-Assitive-Aid-for-Visually-Impaired-People
The Project is a Thesis for Master of Sciene in Computer Science at Lamar University. The main goal of the project is to aid visually impaired people navigate their surroundings and offer text-recognition functionality to detect text such as room numbers etc. Our device consists of using Raspberry Pi 4, Ultrasonic Sensors, LED, Buzzer and Raspberry Pi Camera Module v2 to perform obstacle detection, object recognition and optical character recognition. The results were consolidated and are available to see.
---------------------------------------------------------------------------------------------------------------
Hardware: 
1) Raspberry Pi 4 Model B,
2) HC-SR04 Ultrasonic Sensor,
3) Raspberry Pi Camera Module v2,
4) LED, Buzzer and
5) a 5v/2A Power Bank. (5v/3A can also be used but not more than that).
----------------------------------------------------------------------------------------------------------------
Software:
1) Operating System: Raspberry Pi OS, optimized for performance on the Pi hardware.
2) Open CV: For processing images captured by the camera for recognition tasks.
3) TensorFlow Lite: Runs our deep learning models for object detection and text recognition.
4) GPIO Library: Manages input and output operations on the Raspberry Pi, interfacing with the hardware components like sensors and indicators.
5) Google TTS and Pytesseract: Convert text detected by the OCR into spoken words using text-to-speech technology, making information accessible audibly.
-----------------------------------------------------------------------------------------------------------------
Install packages for OpenCV, TensorFlow Lite, Pygame, Pillow (PIL), Pytesseract, Google TTS.
------------------------------------------------------------------------------------------------------------------
