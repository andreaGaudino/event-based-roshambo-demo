import dv_processing as dv
import cv2
import sys
import time
import threading
from datetime import timedelta
import numpy as np
from utils import IMSIZE, classify_img, PRED_TO_SYMBOL, WINNING_MOVES


cv2.namedWindow("prova", cv2.WINDOW_NORMAL)
TCP_IP = "127.0.0.1"
TCP_PORT = 7777
SCREEN_W, SCREEN_H = 1000, 800  
screen = np.zeros((int(SCREEN_H), int(SCREEN_W), 3), dtype=np.uint8)

try:
    print(f"Attempting to connect to DV TCP Server at {TCP_IP}:{TCP_PORT}...")
    # Replace standard camera capture with the TCP NetworkReader
    capture = dv.io.NetworkReader(TCP_IP, TCP_PORT)
    if not capture.isEventStreamAvailable():
        print("The camera is not returning a stream of events.")
        sys.exit(1)
    print("Successfully connected to the TCP stream!")
    CAMERA_ON = True
except Exception as e:
    print(f"Error connecting to the TCP stream: {e}")
    print("Falling back to offline mode. Ensure DV GUI is running and the TCP Server node is active.")

resolution = capture.getEventResolution()
visualizer = dv.visualization.EventVisualizer(resolution)

# Pre-calcolo dimensioni UI
img_size = int(SCREEN_H * 0.6)
img_x = int(SCREEN_W / 20)
img_y = int(SCREEN_H / 15)
winning_img_x = int(SCREEN_W - img_size - int(SCREEN_W / 20))



running = True


def visualize_frame(events):
    global running
    if not running or events.size() == 0:
        return

    screen.fill(0)
    frame = visualizer.generateImage(events)
    
    # Conversione e preprocessing visivo (veloce)
    if frame.dtype != np.uint8:
        img = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else np.clip(frame, 0, 255).astype(np.uint8)
    else:
        img = frame

    # 1. Resize the raw RGB image to the UI size
    raw_cam_view = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # 2. If 'img' is BGR (OpenCV default), convert to RGB to match your 'screen' logic
    # If 'img' is already RGB, you can skip this conversion
    raw_cam_view_rgb = cv2.cvtColor(raw_cam_view, cv2.COLOR_BGR2RGB)

    # 3. Paste it onto the screen at the pre-calculated coordinates
    # Position: [y_start : y_end, x_start : x_end]
    screen[img_y:img_y + img_size, img_x:img_x + img_size] = raw_cam_view_rgb
    

    cv2.imshow("prova", screen)

slicer = dv.EventStreamSlicer()
slicer.doEveryTimeInterval(timedelta(milliseconds=33), visualize_frame)

print("Start reading in real time. Type 'q' to interrupt.")

try:
    while capture.isRunning() and running:
        events = capture.getNextEventBatch()
        if events is not None and events.size() > 0:
            slicer.accept(events)
        else:
            time.sleep(0.001)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            
        try:
            if cv2.getWindowProperty("prova", cv2.WND_PROP_VISIBLE) < 1:
                running = False
        except Exception:
            pass

except KeyboardInterrupt:
    print("\nReading interrupted.")
finally:
    running = False # Avvisa il thread AI di spegnersi
    cv2.destroyAllWindows()