import pygame
import cv2
import numpy as np
import glob
import os
from utils import classify_img, majority_vote, IMSIZE, PRED_TO_SYMBOL, WINNING_MOVES




def run_offline_mode(screen, clock, interpreter, input_details, output_details, 
                     voter, font_title, font_text, font_small, winning_imgs, 
                     SCREEN_W, SCREEN_H):
    
    image_paths = sorted(glob.glob(os.path.join("sample_frames", "*.png")))
    if not image_paths:
        print("Error: image not found in sample_frames")
        return

    print(f"Found {len(image_paths)} images for testing. Starting PyGame...")
    
    running = True

    # --- GAME LOOP ---
    for img_path in image_paths:
        if not running:
            break
        
        # 1. Handling quitting events (Click on 'X' on the window or typing 'Q')
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        # 2. Reading image with OpenCV (format needed by the NN)
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        
        img_resized = cv2.resize(frame, (IMSIZE, IMSIZE), interpolation=cv2.INTER_AREA)
        
        # 3. Prediction phase
        pred_name, pred_idx, pred_vector = classify_img(img_resized, interpreter, input_details, output_details)
        final_vote_idx = voter.new_prediction_and_vote(pred_idx)
        
        # 4. GRAPHIC RENDER 
        screen.fill((40, 40, 40))   # filling the background with grey
        
        # We dispaly the flow of images on the left 
        img_size = int(SCREEN_H*0.65)
        cam_view = cv2.resize(img_resized, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        cam_view_color = cv2.cvtColor(cam_view, cv2.COLOR_GRAY2RGB)
        
        # Pygame requires the image axis transposed
        cam_view_transposed = np.transpose(cam_view_color, (1, 0, 2))
        cam_surface = pygame.surfarray.make_surface(cam_view_transposed)
        img_x = int(SCREEN_W/20)
        img_y = int(SCREEN_H/15)
        screen.blit(cam_surface, (img_x, img_y)) 
        
        # Writing the name of the file beneath the img
        txt_file = font_small.render(f"File: {os.path.basename(img_path)}", True, (150, 150, 150))
        screen.blit(txt_file, (img_x, img_y + img_size + 10))

        if final_vote_idx is not None:
            confirmed_move = PRED_TO_SYMBOL[final_vote_idx]
            winning_move = WINNING_MOVES[confirmed_move]
            
            # Testi risultato
            txt_you = font_title.render(f"Your move: {confirmed_move.upper()}", True, (255, 255, 255))
            screen.blit(txt_you, (img_x, 10))
            
            txt_model = font_title.render(f"Model move: {winning_move.upper()}", True, (255, 200, 255))
            winning_img_x = img_x + img_size + int(img_size)/10
            screen.blit(txt_model, (winning_img_x, 10))

            # Display the winning image
            if winning_move in winning_imgs:
                winning_surface = winning_imgs[winning_move]
                winning_img_resized = pygame.transform.smoothscale(winning_surface, (img_size, img_size))
                screen.blit(winning_img_resized, (winning_img_x, img_y))

            # Probabilities
            txt_prob = font_text.render("Outcome probabilities:", True, (255, 255, 255))
            text_rect = screen.blit(txt_prob, (img_x, img_y + img_size + 40))
            left, top = text_rect.topleft
            right, bottom = text_rect.bottomright
            
            start_y = img_y + img_size + 80
            for i in range(len(pred_vector)):
                label = font_small.render(f"{PRED_TO_SYMBOL[i].upper()}", True, (200, 200, 200))
                screen.blit(label, (img_x, start_y + i*40))
                
                # Calcola la lunghezza della barra
                bar_w = int(400 * float(pred_vector[i]))
                color = (0, 200, 100) if bar_w > 15 else (200, 50, 50)
                
                # Disegna rettangolo barra
                pygame.draw.rect(screen, color, (int(right/2) + 20, start_y + i*40, bar_w, 20))

        # Aggiorna fisicamente lo schermo
        pygame.display.flip()
        
        # Forza il ciclo a girare a ~30 FPS (Frames Per Second)
        clock.tick(20)    