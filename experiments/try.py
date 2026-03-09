import os
import numpy as np
import cv2

# Define input and output directories
input_dir = os.path.join('datasets', 'segmented2(txt)')
output_dir = os.path.join('datasets', 'segmented2(img)')

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of subfolders to process
subfolders = ['healthy', 'sick']

for subfolder in subfolders:
    input_subfolder = os.path.join(input_dir, subfolder)
    output_subfolder = os.path.join(output_dir, subfolder)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)
    
    count = 0  # count of converted image files in this subfolder

    # Process each text file in the subfolder
    for fname in os.listdir(input_subfolder):
        if fname.lower().endswith('.txt'):
            input_path = os.path.join(input_subfolder, fname)
            # Save with the same base name but a .png extension
            base_name = os.path.splitext(fname)[0]
            output_path = os.path.join(output_subfolder, base_name + '.png')
            
            try:
                # Read the file and detect the delimiter (assume ";" if present)
                with open(input_path, 'r') as file:
                    first_line = file.readline().strip()
                    delimiter = ";" if ";" in first_line else None
                # Load the image data from the text file
                img = np.loadtxt(input_path, delimiter=delimiter)
            except Exception as e:
                print(f"Error reading {input_path}: {e}")
                continue

            # Normalize and convert to uint8 if image is 2D or 3D
            if img.ndim == 2:
                img_min, img_max = img.min(), img.max()
                if img_max - img_min != 0:
                    img_norm = (img - img_min) / (img_max - img_min) * 255
                else:
                    img_norm = np.zeros_like(img)
                img_uint8 = img_norm.astype(np.uint8)
            elif img.ndim == 3:
                # If the file already contains a channel dimension (e.g. shape (H,W,1)), squeeze it.
                if img.shape[-1] == 1:
                    img = np.squeeze(img, axis=-1)
                img_min, img_max = img.min(), img.max()
                if img_max - img_min != 0:
                    img_norm = (img - img_min) / (img_max - img_min) * 255
                else:
                    img_norm = np.zeros_like(img)
                img_uint8 = img_norm.astype(np.uint8)
            else:
                # For any unexpected dimensions, attempt a direct conversion.
                img_uint8 = img.astype(np.uint8)
            
            # Write the image using cv2.imwrite (by default, writes grayscale if single channel)
            success = cv2.imwrite(output_path, img_uint8)
            if success:
                count += 1
                print(f"Converted {input_path} -> {output_path}")
            else:
                print(f"Failed to save image for {input_path}")
    
    print(f"Total converted images in {subfolder}: {count}")
