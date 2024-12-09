import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tifffile import imread

# Define the data range, assuming images are scaled between 0 and 1
DATA_RANGE = 1.0

# Function to calculate 2D SSIM for each spatial slice
def calculate_2d_ssim(ground_truth, generated):
    ssim_values = [ssim(ground_truth[i], generated[i], data_range=ground_truth.max() - ground_truth.min()) for i in range(ground_truth.shape[0])]
    return np.mean(ssim_values)

# Function to calculate 3D SSIM for the full spectral cube
def calculate_3d_ssim(ground_truth, generated):
    return ssim(ground_truth, generated, data_range=ground_truth.max() - ground_truth.min(), multichannel=True)

# Initialize dictionary to store results
results = {'Patch Size': [], 'Average 2D SSIM': [], 'Average 3D SSIM': []}

# Directory structure (modify this as needed)
base_dir = 'test\\patch_test_2'
#base_dir = "/uufs/chpc.utah.edu/common/home/u1528328/Documents/MK_NASA_HSI_ML/test/patch_test_2"
# Loop through folders representing different patch sizes
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Initialize SSIM accumulators for the folderx
    ssim_2d_values, ssim_3d_values = [], []
    
    # Loop through 8 image sets within each folder
    for i in range(8):
        ground_truth = imread(os.path.join(folder_path, f"cb_raw_{i}.tif"))  # Ground truth HSI
        generated = imread(os.path.join(folder_path, f"tl_gen_{i}.tif"))     # Generated HSI

        # Calculate 2D and 3D SSIM for this pair
        ssim_2d = calculate_2d_ssim(ground_truth, generated)
        ssim_3d = calculate_3d_ssim(ground_truth, generated)
        
        # Append SSIM values
        ssim_2d_values.append(ssim_2d)
        ssim_3d_values.append(ssim_3d)
    
    # Calculate average SSIM for the folder
    avg_ssim_2d = np.mean(ssim_2d_values)
    avg_ssim_3d = np.mean(ssim_3d_values)
    
    # Store results in the dictionary
    results['Patch Size'].append(folder)
    results['Average 2D SSIM'].append(avg_ssim_2d)
    results['Average 3D SSIM'].append(avg_ssim_3d)

# Convert results to a DataFrame and display
# Test for SSIM correctness
test_image = np.random.rand(106, 120, 120).astype(np.float32)  # Random test image

# SSIM should be 1 when comparing the same image
ssim_2d_test = calculate_2d_ssim(test_image, test_image)
ssim_3d_test = calculate_3d_ssim(test_image, test_image)

print(f"2D SSIM (self-comparison): {ssim_2d_test}")  # Expect 1.0
print(f"3D SSIM (self-comparison): {ssim_3d_test}")  # Expect 1.0

results_df = pd.DataFrame(results)
print(results_df)
