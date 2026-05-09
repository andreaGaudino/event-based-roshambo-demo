import time
import threading
import queue
import dv_processing as dv
import cv2
from datetime import timedelta, datetime
import numpy as np
from utils import IMSIZE, classify_img, PRED_TO_SYMBOL, WINNING_MOVES


def run_reading_camera_live(capture, camera_name, screen, interpreter, input_details, output_details, voter, winning_imgs, SCREEN_W, SCREEN_H, stats_file):    
    resolution = capture.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    img_size = int(SCREEN_H*0.6)
    img_x = int(SCREEN_W/20)
    img_y = int(SCREEN_H/15)
    winning_img_x = SCREEN_W - img_size - int(SCREEN_W/20)

    processed_winning_imgs = {}
    winning_masks = {}

    for move, surface in winning_imgs.items():
        # Resize once
        resized = cv2.resize(surface, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        
        # Pre-calculate masks once
        if resized.ndim == 3 and resized.shape[2] == 4:
            # Has alpha channel
            b, g, r, a = cv2.split(resized)
            processed_winning_imgs[move] = cv2.merge([b, g, r])
            winning_masks[move] = a > 10 # Create boolean mask from alpha
        else:
            # No alpha channel: chroma-key the background once
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            processed_winning_imgs[move] = resized
            bg = resized[0, 0].astype(int)
            diff = np.linalg.norm(resized.astype(int) - bg, axis=2)
            winning_masks[move] = diff > 30 # Create boolean mask from color difference

    running = True
    start_time = None
    print('starting clock')
    previous_move = ''

    

    def visualize_frame(events):
        nonlocal running, previous_move, start_time
        screen.fill(0)
        if events.size() > 0:
            frame = visualizer.generateImage(events)
            if not start_time:
                start_time = events.getLowestTime() / 1e6
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    img = (frame * 255).astype(np.uint8)
                else:
                    img = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                img = frame

            # Cinverting in greyscale
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Inverting black and white to have black on the background
            inverted = 255 - gray
            # Binary treshold to get defined images (removing noise)
            #_, bw_inv = cv2.threshold(inverted, 20, 255, cv2.THRESH_BINARY)

            resized_img = cv2.resize(inverted, (IMSIZE, IMSIZE), interpolation=cv2.INTER_AREA)
            pred_name, pred_idx, pred_vector = classify_img(resized_img, interpreter, input_details, output_details)
            final_vote_idx = voter.new_prediction_and_vote(pred_idx)
            prediction_time = events.getLowestTime() / 1e6
            
            # cam_view = cv2.resize(resized_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            # cam_view_color = cv2.cvtColor(cam_view, cv2.COLOR_GRAY2RGB)
            # # Place cam view into screen (rows=y, cols=x)
            # screen[img_y:img_y + img_size, img_x: img_x + img_size] = cam_view_color


            raw_cam_view = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            #raw_cam_view_rgb = cv2.cvtColor(raw_cam_view, cv2.COLOR_BGR2RGB)
            screen[img_y:img_y + img_size, img_x:img_x + img_size] = 255 - raw_cam_view

            if final_vote_idx is not None:
                confirmed_move = PRED_TO_SYMBOL[final_vote_idx]
                winning_move = WINNING_MOVES[confirmed_move]
                
                if winning_move != previous_move:
                    print(f'{winning_move}, {prediction_time - start_time}', file=stats_file)
                    previous_move = winning_move


                color = (255, 255, 255)
                # Testing results
                txt_you = f"Your move: {confirmed_move.upper()}"
                cv2.putText(screen, txt_you, (img_x, img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
                

                txt_model = f"Model move: "
                (text_width, text_height), _ = cv2.getTextSize(txt_model, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.putText(screen, txt_model, (int(winning_img_x), img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if winning_move.upper() == 'ROCK':
                    color = (0, 0, 255)
                elif winning_move.upper() == 'PAPER':
                    color = (0, 255, 0)
                elif winning_move.upper() == 'SCISSORS':
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)
                cv2.putText(screen, winning_move.upper(), (int(winning_img_x) + text_width, img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


                # Display the winning image (robust: handle alpha channel and out-of-bounds)
                if winning_move in processed_winning_imgs:
                    win_img = processed_winning_imgs[winning_move]
                    win_mask = winning_masks[winning_move]
                    
                    y1 = int(img_y)
                    y2 = y1 + img_size
                    x1 = int(winning_img_x)
                    x2 = x1 + img_size
                    
                    # Grab the background slice from the screen
                    dst = screen[y1:y2, x1:x2]
                    
                    # Instantaneous numpy copy using the pre-calculated mask!
                    dst[win_mask] = win_img[win_mask]
                    
                    # Put it back on the screen
                    screen[y1:y2, x1:x2] = dst

                # Probabilities
                txt_prob = "Outcome probabilities:"
                (text_width, text_height), _ = cv2.getTextSize(PRED_TO_SYMBOL[3].upper(), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)  
                cv2.putText(screen, txt_prob, (img_x, img_y + img_size + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                right = img_x + text_width 
                
                start_y = img_y + img_size + 80 +  text_height
                for i in range(4):
                    label = f"{PRED_TO_SYMBOL[i].upper()}"
                    cv2.putText(screen, label, (img_x, start_y + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1)
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    # bar lenght
                    bar_w = int(400 * float(pred_vector[i]))
                    color = (0, 200, 100) if bar_w > 15 else (50, 50, 200)
                    cv2.rectangle(screen, (right + 20, start_y + i*40 - label_height), (right + 20 + bar_w, start_y + i*40 + 10), color, -1)


            cv2.imshow(camera_name, screen)


            
    slicer = dv.EventStreamSlicer()
    # The slicer calls visualize_frame every 33ms (30 FPS)
    # 33ms is probably not enough, trying with 40ms 
    slicer.doEveryTimeInterval(timedelta(milliseconds=33), visualize_frame)

    print("Start reading in real time. Type 'q' to interrupt.")

    # Create a queue that only holds the 5 most recent batches
    event_queue = queue.Queue(maxsize=5)

    def event_reader_thread():
        """Reads events from the camera as fast as possible in the background."""
        while capture.isRunning() and running:
            events = capture.getNextEventBatch()
            if events is not None and events.size() > 0:
                # If the queue is full (processing is too slow), drop the oldest batch
                if event_queue.full():
                    try:
                        event_queue.get_nowait()
                    except queue.Empty:
                        pass
                event_queue.put(events)
            else:
                time.sleep(0.001)

    # Start the background reading thread
    reader = threading.Thread(target=event_reader_thread)
    reader.daemon = True
    reader.start()

    # Main thread handles the processing and UI
    try:
        while capture.isRunning() and running:
            try:
                # Grab the latest events from our queue
                events = event_queue.get(timeout=0.1)
                
                # The slicer will now process the events, and jump forward in time 
                # if older events were dropped by the queue.
                slicer.accept(events)
            except queue.Empty:
                pass # Queue is empty, just wait for the next loop

            # Handle UI interrupts
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            try:
                if cv2.getWindowProperty(camera_name, cv2.WND_PROP_VISIBLE) < 1:
                    running = False
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\nReading interrupted.")
    finally:
        running = False # Ensure the background thread shuts down
        cv2.destroyAllWindows()