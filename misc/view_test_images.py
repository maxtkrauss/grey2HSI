import matplotlib.pyplot as plt
import numpy as np
import tifffile
import os
from src.utils import reconstruct_rgb  # Custom utility for RGB reconstruction

from torchmetrics import MeanSquaredError  # Import for error metrics
from matplotlib.patches import Rectangle  # Import Rectangle for drawing patches
from sklearn.metrics import r2_score

# Folder paths for images
img_folder = 'test/pol_0_test/'

# Figure title
title = "Polarization (0 degrees) reconstructed (46x46 patch-size model)"

# Load images from the respective folders
tl_imgs = np.array([tifffile.imread(img_folder + f) for f in os.listdir(img_folder) if f.endswith(".tif") and f.startswith("tl_raw")][:4])
cb_imgs = np.array([tifffile.imread(img_folder + f) for f in os.listdir(img_folder) if f.endswith(".tif") and f.startswith("cb_raw")][:4])
gen_imgs = np.array([tifffile.imread(img_folder + f) for f in os.listdir(img_folder) if f.endswith(".tif") and f.startswith("tl_gen")][:4])

# Define wavelengths range
wavelengths = np.linspace(450, 800, 93)

# Define the ROI coordinates
x1, x2, y1, y2 = 20, 40, 20, 40

def spectra_comparison(x1, x2, y1, y2):
    # Extract the reflectance values for the selected area
    selected_area_cb = cb_img[:, y1:y2+1, x1:x2+1]
    selected_area_gen = gen_img[:, y1:y2+1, x1:x2+1]

    # Calculate the average reflectance values for the selected area
    area_val_cb = np.mean(selected_area_cb, axis=(1, 2))
    area_val_gen = np.mean(selected_area_gen, axis=(1, 2))

    # Store the normalized reflectance values
    spectra_cb = area_val_cb
    spectra_gen = area_val_gen

    return(spectra_gen, spectra_cb) 

# Display shape of generated images for sanity check
print(f"Generated images shape: {gen_imgs.shape}")
print(f"Ground truth images shape: {cb_imgs.shape}")

# Set up the plot grid with 4 rows and 1 column
fig, ax = plt.subplots(4, 3, figsize=(5, 14))

for i in range(3):
# Iterate over the first image for comparison
    cb_img, gen_img, tl_img = cb_imgs[i], gen_imgs[i], tl_imgs[i]

    # Compute error metrics
    mae = np.mean(np.abs(cb_img - gen_img), axis=(1, 2))
    rmse = np.sqrt(np.mean((cb_img - gen_img)**2, axis=(1, 2)))
    rase = np.sqrt(np.mean(rmse**2)) * 100 / np.mean(cb_img)

    rrmse_image = np.mean(rmse)  # Relative RMSE for the image
    rmae_image = np.mean(mae)    # Relative MAE for the image

    fig.suptitle(title)

    sre_alt = np.mean(np.abs(cb_img - gen_img)) / np.mean(cb_img) * 100

    spectra_gen, spectra_cb = spectra_comparison(50, 100, 50, 100)
    avrg_spectra_gen, avrg_spectra_cb = spectra_comparison(0, 119, 0, 119)

    # Plot diffractogram, ground truth HS, and generated HS
    ax[0,i].imshow(tl_img[0])
    ax[0,i].set_title("Diffractogram", fontsize=10)
    ax[0,i].axis('off')

    ax[1,i].imshow(cb_img[50])
    ax[1,i].set_title("Ground Truth HS", fontsize=10)
    ax[1,i].axis('off')

    # Add rectangle to the ground truth HS image
    # rect_cb = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
    # ax[1].add_patch(rect_cb)

    ax[2,i].imshow(gen_img[50])
    ax[2,i].set_title("Generated HS", fontsize=10)
    ax[2,i].axis('off')

    # # Add rectangle to the generated HS image
    # rect_gen = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='red', facecolor='none')
    # ax[2].add_patch(rect_gen)

    # Calculate the R² value
    r2 = r2_score(spectra_cb, spectra_gen)
    avrg_r2 = r2_score(avrg_spectra_cb, avrg_spectra_gen)

    # Plot the spectral comparison with main labels for the curves
    gen_line, = ax[3,i].plot(wavelengths, avrg_spectra_gen, label='Generated', color='b', lw=2)
    cb_line, = ax[3,i].plot(wavelengths, avrg_spectra_cb, label='Ground truth', color='g', lw=2)

    ax[3,i].legend([gen_line, cb_line], [f'Generated', 'Ground truth'], fontsize=8, loc='upper right')

    # Set labels and title
    ax[3,i].set_xlabel('Wavelength (nm)')
    ax[3,i].set_ylabel('Normalized Reflectance')
    ax[3,i].set_title(f"RASE = {rase:.2f}, MAE = {rmae_image:.4f}, RMSE = {rrmse_image:.4f}")

# Adjust spacing between subplots for better readability
plt.subplots_adjust(hspace = 0.5)
plt.show()
