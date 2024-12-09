import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from RGB.HSI2RGB import HSI2RGB
import os

# Load hyperspectral image and apply HSI2RGB transformation
def load_and_convert_image(image_path):
    # Load image as a NumPy array
    img = tiff.imread(image_path)  # Assume image is in (106, x, y) format

    # Normalize if needed
    img = img * 0.5 + 0.5  # Adjust normalization as necessary
    img_np = img.transpose(1, 2, 0)  # Change to (x, y, 106) for HSI2RGB

    # Define wavelengths from 450 to 850 nm for 106 bands
    wl = np.linspace(450, 850, 106)

    # Reshape and apply HSI2RGB
    data = np.reshape(img_np, [-1, img_np.shape[2]])
    RGB = HSI2RGB(wl, data, img_np.shape[0], img_np.shape[1], 65, 0.002)
    
    return RGB, img, wl

# Main function to load and plot HSI image
def plot_hsi_with_rgb_and_selected_bands(image_dir):
    # Load the HSI image (Cubert), diffractogram (TL), and undiffracted image
    cb_file = next(f for f in os.listdir(image_dir) if f.endswith(".tif") and f.startswith("cb_raw"))
    tl_file = next(f for f in os.listdir(image_dir) if f.endswith(".tif") and f.startswith("tl_raw_1"))
    undiffracted_file = next(f for f in os.listdir(image_dir) if f.endswith(".tif") and f.startswith("tl_raw_2"))
    cb_path = os.path.join(image_dir, cb_file)
    tl_path = os.path.join(image_dir, tl_file)
    undiffracted_path = os.path.join(image_dir, undiffracted_file)
    
    # Convert hyperspectral to RGB
    rgb_image, hsi_image, wavelengths = load_and_convert_image(cb_path)
    
    # Load the diffractogram and undiffracted images (use only the first band if needed)
    diffractogram = tiff.imread(tl_path)[0]  # Using only the first band of tl_raw_1
    undiffracted_image = tiff.imread(undiffracted_path)[0]  # Using only the first band of tl_raw_2

    # Define desired target wavelengths
    target_wavelengths = [450, 550, 600, 700, 750]
    selected_indices = [np.argmin(np.abs(wavelengths - wl)) for wl in target_wavelengths]
    actual_wavelengths = [wavelengths[idx] for idx in selected_indices]  # Get actual wavelengths for the selected indices

    # Display the RGB reconstruction, diffractogram, and undiffracted image in the first figure
    fig1, ax1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle("RGB Reconstruction, Diffractogram, and Undiffracted Image", fontsize=16)
    
    # RGB Reconstruction
    ax1[0].imshow(rgb_image)
    ax1[0].set_title("RGB Reconstruction", fontsize=12)
    ax1[0].axis('off')
    
    # Diffractogram
    ax1[1].imshow(diffractogram, cmap='gray')
    ax1[1].set_title("Diffractogram", fontsize=12)
    ax1[1].axis('off')
    
    # Undiffracted Image
    ax1[2].imshow(undiffracted_image, cmap='gray')
    ax1[2].set_title("Undiffracted Image", fontsize=12)
    ax1[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Display the selected wavelengths in a second figure
    fig2, ax2 = plt.subplots(1, 5, figsize=(20, 6))
    fig2.suptitle("Selected Wavelengths from Hyperspectral Image", fontsize=16)

    # Display the image for each selected wavelength with the actual wavelength label
    for i, (idx, actual_wl) in enumerate(zip(selected_indices, actual_wavelengths)):
        ax2[i].imshow(hsi_image[idx], cmap='viridis')
        ax2[i].set_title(f"{actual_wl:.1f} nm", fontsize=12)
        ax2[i].axis('off')

    plt.tight_layout()
    plt.show()

# Usage
image_directory = "test/patch_test_2/pol_0_reinforced_test"  # Set this to your folder path
plot_hsi_with_rgb_and_selected_bands(image_directory)
