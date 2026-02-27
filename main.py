import cv2
import numpy as np
import time
import glob
import os
import ctypes
import tensorflow as tf

# --- 1. Constants ---
IMSIZE = 64 # Image size requested by the Neural Network



CAMERA_ON = False  # Initially set on False, if the camera is available it will be set aìto True

# Map from class output to string prediction
PRED_TO_SYMBOL = {
    0: 'paper', 
    1: 'scissors', 
    2: 'rock', 
    3: 'background'
}

# Game logic: winning classes based on the move of the hand
WINNING_MOVES = {
    'rock': 'paper',
    'paper': 'scissors',
    'scissors': 'rock',
    'background': 'none'
}



# --- 2. Original logic, taken from the original repository  ---

class majority_vote:
    """Filter cmd with majority vote"""
    def __init__(self, window_length, num_classes):
        """Does median filter majority vote over past predictions of human hand symbol
        :param window length: the size of window in votes
        :param num_classes: the number of possible values, 0 to num_classes-1
        :return: the majority if there is one, else None
        """
        self.window_length = window_length
        self.num_classes = num_classes
        self.ptr = 0 # pointer to circular buffer
        self.cirbuf = np.full(self.window_length, -1, dtype=np.int8) # cirular buffer of most recent predictions
        self.cmdcnts = np.zeros(num_classes, dtype=np.int8) # hold the number of votes for each prediction
        self.num_predictions=0

    def new_prediction_and_vote(self, symbol): # cmd is the new value, in range 0 to num_classes-1
        """ Takes new prediction of symbol, returns possible new vote
        :param symbol: the new classification of hand symbol
        :returns: the majority vote or None if there is no majority
        """
        if 0 <= symbol < self.num_classes:
            self.num_predictions+=1
            idx = self.ptr # pointer to current idx in circular buffer
            if self.num_predictions>self.window_length: 
                self.cmdcnts[self.cirbuf[idx]] -= 1 # decrement count for previous prediction but only if we already filled the buffer, otherwise we end up with negative background
            self.cirbuf[idx] = symbol # store latest prediction
            self.cmdcnts[symbol] += 1  # vote for this prediction

            self.ptr = (self.ptr + 1) % self.window_length # increment and wrap pointer

        return self.vote()

    def vote(self): 
        """ produces the majority vote
        :returns: the majority if there is one, otherwise None
        """
        majority_count = self.window_length // 2 + 1 # e.g. 3 for window_length=5
        imax = np.argmax(self.cmdcnts)
        if self.cmdcnts[imax] >= majority_count:
            return imax
        return None

def classify_img(img: np.array, interpreter, input_details, output_details):
    """ Classify uint8 img

    :param img: input image as unit8 np.array range 0-255
    :param interpreter: the TFLITE interpreter
    :param input_details: the input details of interpreter
    :param output_details: the output details of interpreter

    :returns: symbol ('background' 'rock','scissors', 'paper'), class number (0-3), softmax output vector [4]
    """
    input_data = (1/256.) * np.array(np.reshape(img, [1, IMSIZE, IMSIZE, 1]), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    pred_vector = interpreter.get_tensor(output_details[0]['index'])[0]
    pred_idx = np.argmax(pred_vector)
    pred_class_name = PRED_TO_SYMBOL[pred_idx]
    
    return pred_class_name, pred_idx, pred_vector

# --- 3. Game main abilities ---

def main():
    interpreter = tf.lite.Interpreter(model_path="model/dextra_roshambo.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    voter = majority_vote(window_length=5, num_classes=4)


    if not CAMERA_ON:
        # 1. We look for all the images from sample_frames sorted by their name_id
        # glo.glob is able to look for all files using the '*' notation
        image_paths = sorted(glob.glob(os.path.join("sample_frames", "*.png")))
        
        if not image_paths:
            print("Error: images not found in the folder sample_frames")
            return

        print(f"Found {len(image_paths)} test images")
        print("Type 'q' to quit the execution")

        # 2. Iterating on each image (simulating the flow of time)
        for img_path in image_paths:
            
            # Read the image, forcing it to be in grey scale (as seen by the NN)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if frame is None:
                continue
                
            # Resizing to 64x64
            img_resized = cv2.resize(frame, (IMSIZE, IMSIZE), interpolation=cv2.INTER_AREA)
            
            # Prediction phase
            pred_name, pred_idx, pred_vector = classify_img(img_resized, interpreter, input_details, output_details)
            
            # Voting phase
            final_vote_idx = voter.new_prediction_and_vote(pred_idx)
            
            # --- Updating GUI ---
            # create a 3-channel BGR display sized to the current screen resolution
            try:
                user32 = ctypes.windll.user32
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
            except Exception:
                # fallback to a sensible default
                screen_w, screen_h = 1000, 1000

            display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            # ensure the OpenCV window opens fullscreen
            cv2.namedWindow("TEST - Event-Based Roshambo", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("TEST - Event-Based Roshambo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # Displaying current image resized on the left
            left_size = int(screen_h/2 - 50)
            cam_view = cv2.resize(img_resized, (left_size, left_size), interpolation=cv2.INTER_NEAREST)
            cam_view = cv2.cvtColor(cam_view, cv2.COLOR_GRAY2BGR)
            display[50:50+left_size, 100:100+left_size] = cam_view
            
            # Writing file name for debugging purposes
            cv2.putText(display, f"File: {os.path.basename(img_path)}", (10, screen_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

            if final_vote_idx is not None:
                confirmed_move = PRED_TO_SYMBOL[final_vote_idx]
                winning_move = WINNING_MOVES[confirmed_move]
                winning_image_path = os.path.join('symbols', f'{winning_move}.png')
                # read image with alpha if present so we can preserve transparency
                winning_image = cv2.imread(winning_image_path, cv2.IMREAD_UNCHANGED)
                if winning_image is None:
                    print(f"Warning: winning image not found: {winning_image_path}")
                else:
                    win_size = left_size
                    win = cv2.resize(winning_image, (win_size, win_size), interpolation=cv2.INTER_AREA)
                    y1 = int(screen_h/2) + 50
                    x1 = 100
                    y2 = min(y1 + win_size, screen_h)
                    x2 = min(x1 + win_size, screen_w)
                    h = y2 - y1
                    w = x2 - x1
                    win = win[0:h, 0:w]

                    # If image has alpha channel (4th channel), composite using alpha
                    if win.ndim == 3 and win.shape[2] == 4:
                        b, g, r, a = cv2.split(win)
                        alpha = (a.astype(float) / 255.0)[..., None]
                        src_rgb = cv2.merge([b, g, r]).astype(float)
                        dst_rgb = display[y1:y2, x1:x2].astype(float)
                        comp = src_rgb * alpha + dst_rgb * (1.0 - alpha)
                        display[y1:y2, x1:x2] = comp.astype(np.uint8)
                    else:
                        # No alpha: attempt a simple chroma-key to ignore uniform background
                        if win.ndim == 2:
                            win = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
                        # take top-left pixel as background sample
                        bg = win[0, 0].astype(int)
                        diff = np.linalg.norm(win.astype(int) - bg, axis=2)
                        mask = diff > 30  # threshold; tweak if needed
                        # copy only pixels where mask is True
                        dst = display[y1:y2, x1:x2]
                        for c in range(3):
                            ch = dst[..., c]
                            ch[mask] = win[..., c][mask]
                            dst[..., c] = ch
                        display[y1:y2, x1:x2] = dst
                
                # Displaying final result
                cv2.putText(display, f"Your move: ", (int(screen_w/2 + 50), 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"{confirmed_move.upper()}", (int(screen_w/2 + 50), 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                cv2.putText(display, f"Model move: ", (int(screen_w/2 + 50), 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(display, f"{winning_move.upper()}", (int(screen_w/2 + 50), 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                cv2.putText(display, f"Outcome probabilities: ", (int(screen_w/2 + 50), 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                height = 350
                for i in range(len(pred_vector)):
                    height += 50
                    cv2.putText(display, f"{PRED_TO_SYMBOL[i]}", (int(screen_w/2 + 50), height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    # ensure integer coordinates for rectangle
                    x1 = int(screen_w/2 + 50)
                    y1 = height + 10
                    x2 = int(int(screen_w/2 + 50) + (int(screen_w*0.9) - int(screen_w/2 + 50)) * float(pred_vector[i].round(2)))
                    y2 = height + 30
                    color = (0, 255, 0)
                    if x2 - x1 < 10: 
                        color = (0, 0, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, -1)
                    height += 50
            cv2.imshow("TEST - Event-Based Roshambo", display)

            # cv2.waitKey(33) blocks the code for 33 milliseconds in order to give time to 
            # draw the interface
            # If you put waitKey(0) it will proceed only after typing any key
            key = cv2.waitKey(40) 
            if key == ord('q'):
                print('Quitting the process...')
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()