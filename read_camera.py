import dv_processing as dv
import cv2
import sys
from datetime import timedelta
import numpy as np
from utils import IMSIZE, classify_img, PRED_TO_SYMBOL, WINNING_MOVES


def run_reading_camera_live(capture, camera_name, screen, interpreter, input_details, output_details, voter, winning_imgs, SCREEN_W, SCREEN_H):
    # try:
    #     capture = dv.io.camera.open()
    #     print(f"Camera [{capture.getCameraName()}] connected!")
    # except Exception as e:
    #     print(f"Error connecting the camera: {e}")
    #     sys.exit(1)

    if not capture.isEventStreamAvailable():
        print("The camera is not returning a stream of events.")
        sys.exit(1)

    resolution = capture.getEventResolution()
    visualizer = dv.visualization.EventVisualizer(resolution)

    


    running = True


    def visualize_frame(events):
        global running
        if events.size() > 0:
            frame = visualizer.generateImage(events)
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

            # Binary treshold to get defined images
            _, bw_inv = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
            resized_img = cv2.resize(bw_inv, (IMSIZE, IMSIZE), interpolation=cv2.INTER_AREA)
            pred_name, pred_idx, pred_vector = classify_img(resized_img, interpreter, input_details, output_details)
            final_vote_idx = voter.new_prediction_and_vote(pred_idx)
            img_size = int(SCREEN_H*0.6)
            cam_view = cv2.resize(resized_img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            cam_view_color = cv2.cvtColor(cam_view, cv2.COLOR_GRAY2RGB)

            img_x = int(SCREEN_W/20)
            img_y = int(SCREEN_H/15)
            # Place cam view into screen (rows=y, cols=x)
            screen[img_y:img_y + img_size, img_x: img_x + img_size] = cam_view_color
            if final_vote_idx is not None:
                confirmed_move = PRED_TO_SYMBOL[final_vote_idx]
                winning_move = WINNING_MOVES[confirmed_move]
                
                # Testing results
                txt_you = f"Your move: {confirmed_move.upper()}"
                cv2.putText(screen, txt_you, (img_x, img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                txt_model = f"Model move: {winning_move.upper()}"
                winning_img_x = SCREEN_W - img_size - int(SCREEN_W/20)
                cv2.putText(screen, txt_model, (int(winning_img_x), img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Display the winning image (robust: handle alpha channel and out-of-bounds)
                if winning_move in winning_imgs:
                    winning_surface = winning_imgs[winning_move]
                    winning_img_resized = cv2.resize(winning_surface, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    y1 = int(img_y)
                    y2 = y1 + winning_img_resized.shape[0]
                    x1 = int(winning_img_x)
                    x2 = x1 + winning_img_resized.shape[1]
                    # If image has alpha channel, convert to RGB
                    if winning_img_resized.ndim == 3 and winning_img_resized.shape[2] == 4:
                        b, g, r, a = cv2.split(winning_img_resized)
                        alpha = (a.astype(float) / 255.0)[..., None]
                        src_rgb = cv2.merge([b, g, r]).astype(float)
                        dst_rgb = screen[y1:y2, x1:x2].astype(float)
                        comp = src_rgb * alpha + dst_rgb * (1.0 - alpha)
                        screen[y1:y2, x1:x2] = comp.astype(np.uint8)
                    else:
                        # No alpha: attempt a simple chroma-key to ignore uniform background
                        if win.ndim == 2:
                            win = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
                        # take top-left pixel as background sample
                        bg = win[0, 0].astype(int)
                        diff = np.linalg.norm(win.astype(int) - bg, axis=2)
                        mask = diff > 30  # threshold; tweak if needed
                        # copy only pixels where mask is True
                        dst = screen[y1:y2, x1:x2]
                        for c in range(3):
                            ch = dst[..., c]
                            ch[mask] = win[..., c][mask]
                            dst[..., c] = ch
                        screen[y1:y2, x1:x2] = dst

                # Probabilities
                txt_prob = "Outcome probabilities:"
                (text_width, text_height), _ = cv2.getTextSize(PRED_TO_SYMBOL[3].upper(), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)  
                cv2.putText(screen, txt_prob, (img_x, img_y + img_size + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                right = img_x + text_width 
                
                start_y = img_y + img_size + 80 +  text_height
                for i in range(len(pred_vector)):
                    label = f"{PRED_TO_SYMBOL[i].upper()}"
                    cv2.putText(screen, label, (img_x, start_y + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1)
                    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                    # bar lenght
                    bar_w = int(400 * float(pred_vector[i]))
                    color = (0, 200, 100) if bar_w > 15 else (50, 50, 200)
                    cv2.rectangle(screen, (right + 20, start_y + i*40 - label_height), (right + 20 + bar_w, start_y + i*40 + 10), color, -1)


            cv2.imshow(camera_name, screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False

            # If the user closed the OpenCV window (clicked the X), stop the loop
            try:
                if cv2.getWindowProperty(screen, cv2.WND_PROP_VISIBLE) < 1:
                    running = False
            except Exception:
                pass

    slicer = dv.EventStreamSlicer()
    # The slicer calls visualize_frame every 33ms (30 FPS)
    slicer.doEveryTimeInterval(timedelta(milliseconds=33), visualize_frame)

    print("Start reading in real time. Type 'q' to interrupt.")

    try:
        while capture.isRunning() and running:
            events = capture.getNextEventBatch()
            if events is not None and events.size() > 0:
                slicer.accept(events)

    except KeyboardInterrupt:
        print("\nReading interrupted.")
    finally:
        cv2.destroyAllWindows()
        