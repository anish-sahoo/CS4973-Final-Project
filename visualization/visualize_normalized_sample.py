import scipy.io as sio
import numpy as np
import cv2
import os
import sys

def save_normalized_sample():
    """
    Loads a sample .mat file from the MPIIGaze Normalized dataset
    and saves the first left/right eye images to disk.
    """
    # Path to a sample file - adjusting path relative to project root
    mat_path = 'data/MPIIGaze/Data/Normalized/p00/day01.mat'
    
    if not os.path.exists(mat_path):
        print(f"Error: {mat_path} not found.")
        print("Please ensure you are running this from the project root and the data exists.")
        return

    print(f"Loading {mat_path}...")
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Failed to load mat file: {e}")
        return
    
    # Structure in MPIIGaze normalized files is usually data -> left -> image
    # loadmat returns a dictionary
    if 'data' not in mat:
        print("Error: 'data' key not found in mat file.")
        print(f"Available keys: {mat.keys()}")
        return
        
    data = mat['data']
    
    # Extract images
    # The dataset code does: left_imgs = np.array(left_struct.image, dtype=np.float32)
    try:
        left_images = data.left.image
        right_images = data.right.image
    except AttributeError:
        print("Error: Could not find left/right image structure in data.")
        return
    
    # Take the first sample, Shape is likely (N, Height, Width)
    if left_images.ndim == 3:
        l_img = left_images[0]
        r_img = right_images[0]
    elif left_images.ndim == 2:
        l_img = left_images
        r_img = right_images
    else:
        print(f"Unexpected image dimensions: {left_images.ndim}")
        return
        
    # The images in the .mat file are already histogram equalized and cropped (normalized)
    # They are stored as pixel values (0-255) usually.
    
    # Ensure it's uint8 for saving
    l_img_uint8 = l_img.astype(np.uint8)
    r_img_uint8 = r_img.astype(np.uint8)
    
    # Save
    output_dir = 'visualization/samples'
    os.makedirs(output_dir, exist_ok=True)
    
    l_out = os.path.join(output_dir, 'normalized_left_eye.png')
    r_out = os.path.join(output_dir, 'normalized_right_eye.png')
    
    cv2.imwrite(l_out, l_img_uint8)
    cv2.imwrite(r_out, r_img_uint8)
    
    print(f"Saved sample images to {output_dir}")
    print(f"Saved {l_out}")
    print(f"Saved {r_out}")
    print(f"Image shape: {l_img_uint8.shape}")

if __name__ == "__main__":
    save_normalized_sample()
