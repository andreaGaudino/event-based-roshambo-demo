import os
import numpy as np
import cv2
import glob
from dv import LegacyAedatFile

def convert_aedat_to_npy(filepath_in, filepath_out, events_per_frame=2000):
    print(f"Processing: {filepath_in}...")
    
    # Resolution of the DAVIS240C camera used for the RoShamBo dataset
    sensor_width = 240
    sensor_height = 180
    
    frames = []
    xs = []
    ys = []
    
    try:
        # Open the legacy file (handles both AEDAT 2.0 and 3.1)
        with LegacyAedatFile(filepath_in) as f:
            for event in f:
                # Extract event coordinates
                xs.append(event.x)
                ys.append(event.y)
                
                # When we reach the configured number of events, create a frame
                if len(xs) == events_per_frame:
                    # Create a black canvas
                    img = np.zeros((sensor_height, sensor_width), dtype=np.float32)

                    # NumPy trick: add 1 at all (y, x) coordinates read in the chunk
                    np.add.at(img, (ys, xs), 1)

                    # Resize for the model
                    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

                    # Normalize between 0 and 1
                    max_val = img_resized.max()
                    if max_val > 0:
                        img_resized = img_resized / max_val

                    frames.append(img_resized)

                    # Clear buffers for the next frame
                    xs.clear()
                    ys.clear()
                    
    except Exception as e:
        # If the legacy file ends unexpectedly or has corrupted trailing bytes, stop cleanly
        print(f"-> End of file reached or interruption ({e}).")
        print(f"-> Successfully recovered {len(frames)} intact frames.")
        
    if not frames:
        print("No frames generated from this file.\n")
        return

    # Convert to array (N, 64, 64) -> (N, 64, 64, 1) for Keras
    frames_np = np.array(frames, dtype=np.float32)
    frames_np = np.expand_dims(frames_np, axis=-1)

    # Save the file
    np.save(filepath_out, frames_np)
    print(f"File saved: {filepath_out} | Array shape: {frames_np.shape}\n")


# ==========================================
# AUTOMATIC EXECUTION
# ==========================================
if __name__ == "__main__":
    # Enter the correct path to your folder
    aedat_dir = os.path.join('..', 'aedat_files')
    files_to_convert = [
        os.path.join(aedat_dir, f)
        for f in os.listdir(aedat_dir)
    ]

    if not files_to_convert:
        print("No .aedat files found in the specified path.")

    for file_in in files_to_convert:
        file_out = file_in.replace('.aedat', '.npy')
        convert_aedat_to_npy(file_in, file_out)

    print("Batch processing completed!")