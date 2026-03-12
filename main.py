import cv2
import numpy as np
import time
import glob
import os
import pygame
import tensorflow as tf
import sys
import ctypes
from test_images import run_offline_mode
from utils import classify_img, majority_vote, IMSIZE, PRED_TO_SYMBOL, WINNING_MOVES, CAMERA_ON




# --- MAIN GAME LOGIC ---

def main():
    # 1. NN initialization
    print("Model loading...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "dextra_roshambo.tflite")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    voter = majority_vote(window_length=5, num_classes=4)

    # 2. PyGame initialization
    pygame.init()
    if sys.platform.startswith('win'):
        try:
            user32 = ctypes.windll.user32
            SCREEN_W = user32.GetSystemMetrics(0)
            SCREEN_H = user32.GetSystemMetrics(1)*0.9
        except Exception:
            SCREEN_W, SCREEN_H = 1000, 1000
    else:
        try:
            import tkinter as tk
            root = tk.Tk()
            SCREEN_W = root.winfo_screenwidth()
            SCREEN_H = root.winfo_screenheight()
            root.destroy()
        except Exception:
            SCREEN_W, SCREEN_H = 1000, 800    
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Event-Based Roshambo Demo")
    
    # Fonts
    font_title = pygame.font.SysFont(None, 60)
    font_text = pygame.font.SysFont(None, 40)
    font_small = pygame.font.SysFont(None, 24)

    # 3. Loading winning moves images
    print("Loading winning images...")
    winning_imgs = {}
    for move in ['rock', 'paper', 'scissors']:
        img_path = os.path.join(base_dir, 'symbols', f'{move}.png')
        if os.path.exists(img_path):
            img_surface = pygame.image.load(img_path).convert_alpha()  #convert_alpha handles images without background
            winning_imgs[move] = img_surface

    clock = pygame.time.Clock() # To keep 30 FSP constant

    if CAMERA_ON:
        pass
    else:
        run_offline_mode(
            screen=screen,
            clock=clock,
            interpreter=interpreter,
            input_details=input_details,
            output_details=output_details,
            voter=voter,
            font_title=font_title,
            font_text=font_text,
            font_small=font_small,
            winning_imgs=winning_imgs,
            SCREEN_W=SCREEN_W,
            SCREEN_H=SCREEN_H
        )
        
 

    pygame.quit()


if __name__ == '__main__':
    main()