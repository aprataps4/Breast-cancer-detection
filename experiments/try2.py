import os
import glob
import cv2
import numpy as np
import random
import shutil
import re

def load_txt_image(file_path):
    """
    Loads image data from a text file.
    Assumes each line corresponds to a row and numbers are whitespace‐separated.
    Returns a numpy array of type float32.
    """
    return np.loadtxt(file_path, dtype=np.float32)

def save_txt_image(file_path, image):
    """
    Saves the image (a numpy array) to a text file.
    Each row is written on a new line with values separated by a space.
    """
    np.savetxt(file_path, image, fmt='%.6f')

def augment_image(image, max_angle=2):
    """
    Augments the image by:
      1. Rotating it by a random angle between -max_angle and max_angle degrees.
      2. Flipping the rotated image horizontally.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Choose a random angle between -max_angle and max_angle.
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate the image.
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Flip horizontally.
    flipped = cv2.flip(rotated, 1)
    return flipped

def make_new_filename(rep_filename, new_seq):
    """
    Given a representative filename (assumed to have the pattern
    T####.rest.txt) create a new filename by replacing the T number with new_seq.
    For example, if rep_filename is:
       T0179.1.1.S.2013-08-16.00.txt
    and new_seq is 180, then the new filename will be:
       T0180.1.1.S.2013-08-16.00.txt
    """
    base, ext = os.path.splitext(rep_filename)
    m = re.match(r'^(T)(\d+)(\..*)$', base)
    if m:
        prefix = m.group(1)          # "T"
        num_str = m.group(2)         # e.g. "0179"
        rest = m.group(3)            # e.g. ".1.1.S.2013-08-16.00"
        new_num_str = str(new_seq).zfill(len(num_str))
        return prefix + new_num_str + rest + ext
    else:
        # If the pattern doesn't match, simply append new_seq.
        return rep_filename + f"_{new_seq}" + ext

def get_max_T_number(folder_path):
    """
    Scans all .txt files in folder_path (assumed to have names starting with T followed by digits)
    and returns the maximum T-number (as an integer) found. Returns -1 if none found.
    """
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    max_T = -1
    for f in files:
        base = os.path.basename(f)
        m = re.match(r'^T(\d+)', base)
        if m:
            try:
                num = int(m.group(1))
                max_T = max(max_T, num)
            except:
                continue
    return max_T

def process_folder(source_folder, dest_folder):
    """
    Processes one folder as follows:
      1. Copies all .txt files from source_folder to dest_folder.
      2. Then, if the total count of files in dest_folder is less than 200, it
         augments images (rotation by ±2° and horizontal flip) from randomly selected files
         until the total file count reaches 200. The new files are named by incrementing
         the T-number in a representative filename.
    """
    # Ensure destination folder exists.
    os.makedirs(dest_folder, exist_ok=True)
    
    # Copy original txt files from source to destination.
    src_files = glob.glob(os.path.join(source_folder, "*.txt"))
    src_files.sort()
    for file_path in src_files:
        shutil.copy2(file_path, dest_folder)
        print(f"Copied original file: {file_path} to {dest_folder}")
    
    # Get list of files in destination folder.
    dest_files = glob.glob(os.path.join(dest_folder, "*.txt"))
    count = len(dest_files)
    
    # Get a representative filename (assume all files follow same pattern).
    rep_filename = os.path.basename(dest_files[0]) if dest_files else "T0000.txt"
    
    # Determine the next T number by scanning the folder.
    max_T = get_max_T_number(dest_folder)
    next_T = max_T + 1
    
    # Augment until total file count is 200.
    while count < 200:
        # Randomly select a file from the existing files.
        file_to_aug = random.choice(dest_files)
        try:
            image = load_txt_image(file_to_aug)
        except Exception as e:
            print(f"Error loading {file_to_aug}: {e}")
            continue
        
        # Ensure image is 2D.
        if image.ndim == 1:
            image = image.reshape(1, -1)
        
        augmented = augment_image(image)
        
        # Generate new filename using the representative filename and next_T.
        new_filename = make_new_filename(rep_filename, next_T)
        next_T += 1
        
        out_file = os.path.join(dest_folder, new_filename)
        save_txt_image(out_file, augmented)
        print(f"Saved augmented file: {out_file}")
        
        # Update list and count.
        dest_files.append(out_file)
        count += 1

def main():
    # Source folders for the original dataset.
    source_healthy = r"C:\Users\2004a\OneDrive\Desktop\BTP\BTP(Breast Cancer using IR)\BTP(Breast cancer using IR)\datasets\segmented(txt)\healthy"
    source_sick    = r"C:\Users\2004a\OneDrive\Desktop\BTP\BTP(Breast Cancer using IR)\BTP(Breast cancer using IR)\datasets\segmented(txt)\sick"
    
    # Destination base folder.
    dest_base = r"C:\Users\2004a\OneDrive\Desktop\BTP\BTP(Breast Cancer using IR)\BTP(Breast cancer using IR)\datasets\segmented2(txt)"
    dest_healthy = os.path.join(dest_base, "healthy")
    dest_sick = os.path.join(dest_base, "sick")
    
    # Create destination folder structure.
    os.makedirs(dest_healthy, exist_ok=True)
    os.makedirs(dest_sick, exist_ok=True)
    
    print("Processing healthy folder...")
    process_folder(source_healthy, dest_healthy)
    print("Processing sick folder...")
    process_folder(source_sick, dest_sick)
    
if __name__ == "__main__":
    main()
