import pygame
import os
import time
music_folder = "music"
pygame.mixer.init()

def play_music_from_folder(music_folder):
  
    music_files = [f for f in os.listdir(music_folder) if f.endswith(('.mp3', '.wav'))]

    if not music_files:
        print("No music files found in the specified folder.")
        return
    for music_file in music_files:
        file_path = os.path.join(music_folder, music_file)
        print(f"Playing: {music_file}")
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)

play_music_from_folder(music_folder)
