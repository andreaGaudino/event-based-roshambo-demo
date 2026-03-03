import pygame
import cv2
import numpy as np
import glob
import os


# --- 1. COSTANTS ---
IMSIZE = 64 # Image size requested by the Neural Network
CAMERA_ON = False

# Predition map
PRED_TO_SYMBOL = {
    0: 'paper', 
    1: 'scissors', 
    2: 'rock', 
    3: 'background'
}

# Winning moves wrt to user moves
WINNING_MOVES = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock',
    'background': 'none'
}

# --- 2.Model logic ---

class majority_vote:
    def __init__(self, window_length, num_classes):
        self.window_length = window_length
        self.num_classes = num_classes
        self.ptr = 0 
        self.cirbuf = np.full(self.window_length, -1, dtype=np.int8)
        self.cmdcnts = np.zeros(num_classes, dtype=np.int8) 
        self.num_predictions = 0

    def new_prediction_and_vote(self, symbol):
        if 0 <= symbol < self.num_classes:
            self.num_predictions += 1
            idx = self.ptr 
            if self.num_predictions > self.window_length: 
                self.cmdcnts[self.cirbuf[idx]] -= 1 
            self.cirbuf[idx] = symbol 
            self.cmdcnts[symbol] += 1  
            self.ptr = (self.ptr + 1) % self.window_length 

        majority_count = self.window_length // 2 + 1 
        imax = np.argmax(self.cmdcnts)
        if self.cmdcnts[imax] >= majority_count:
            return imax
        return None

def classify_img(img: np.array, interpreter, input_details, output_details):
    input_data = (1/256.) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    pred_vector = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(pred_vector)
    pred_class_name = PRED_TO_SYMBOL[pred_idx]
    
    return pred_class_name, pred_idx, pred_vector