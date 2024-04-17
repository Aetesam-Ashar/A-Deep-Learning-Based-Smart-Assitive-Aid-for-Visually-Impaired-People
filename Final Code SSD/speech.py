from gtts import gTTS  # Imports the Google Text-to-Speech library
import os  # Provides a way of using operating system dependent functionality
import pygame  # Imports pygame, a library used for multimedia & game programming

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')  # Creates a TTS object for the given text in English
    tts.save("output.mp3")  # Saves the spoken text to an mp3 file
    play_audio("output.mp3")  # Calls the function to play the audio file
    os.remove("output.mp3")  # Deletes the audio file after playing

def play_audio(file_path):
    pygame.mixer.init()  # Initializes the mixer module in pygame
    pygame.mixer.music.load(file_path)  # Loads the audio file for playing
    pygame.mixer.music.play()  # Starts playing the loaded audio file
    while pygame.mixer.music.get_busy():  # Loops while the audio is still playing
        pygame.time.Clock().tick(10)  # Limits the while loop to checking 10 times per second

if __name__ == "__main__":
    import sys  # Imports the sys module, which provides access to some variables used or maintained by the interpreter
    if len(sys.argv) > 1:  # Checks if any additional command line arguments are provided
        text_to_speech(' '.join(sys.argv[1:]))  # Joins all arguments into a single string and passes it to the text_to_speech function
    else:
        print("No text provided for speech.")  # Prints a message if no arguments were provided