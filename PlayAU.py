import time
import pygame
import threading

pygame.init()
audio_file = 'AU.mp3'
sound = pygame.mixer.Sound(audio_file)
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=8192)
pygame.mixer.music.load(audio_file)

sound_playing = False
last_audio_play_time=0
def play_audio():
    # global sound_playing
    # sound.play()
    # time.sleep(2)  # Adjust the delay based on your audio length
    # pygame.mixer.music.stop()
    # sound_playing = False
    # pygame.mixer.music.play()
    # pygame.time.delay(10000)  # Adjust the sleep duration based on your audio length
    # pygame.mixer.music.stop()
    global sound_playing, last_audio_play_time
    sound.play()
    pygame.time.delay(2000)  # Adjust the delay based on your audio length
    pygame.mixer.music.stop()
    time.sleep(1)
    sound_playing = False
    last_audio_play_time = time.time()  # Update the timer after audio play


def play_audio_threaded():
    thread = threading.Thread(target=play_audio)
    thread.daemon = True
    thread.start()



def stop_audio():
    global sound_playing
    pygame.mixer.music.stop()
    sound_playing = False