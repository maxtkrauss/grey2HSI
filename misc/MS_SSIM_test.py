import torch
from torch import rand  
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

preds = torch.rand([3, 3, 256, 256])
target = preds * 0.75
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
ms_ssim(preds, target)
print(ms_ssim(preds, preds))

import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import tifffile as tiff

# Define the MS-SSIM metric
ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(kernel_size=5, betas=(0.0448, 0.2856, 0.3001))

# Load the images
def load_image(filepath):
    """
    Load a multi-band image using tifffile and convert to a PyTorch tensor.
    """
    image = tiff.imread(filepath)
    # Convert to PyTorch tensor and add channel dimension if needed
    tensor = torch.tensor(image, dtype=torch.float32)
    if len(tensor.shape) == 2:  # If grayscale, add channel dimension
        tensor = tensor.unsqueeze(0)
    return tensor

# File paths for the images
image1_path = 'test\\patch_test_2\\pol_0_reinforced_MSE\\cb_raw_0.tif'
image2_path = 'test\\patch_test_2\\pol_0_reinforced_MSE\\tl_gen_0.tif'

# Load images as tensors
image1 = load_image(image1_path)
image2 = load_image(image2_path)

print("Ground truth shape: ", image1.unsqueeze(0).shape)
print("Generated shape ", image2.unsqueeze(0).shape)

# Ensure the images have the same shape
if image1.shape != image2.shape:
    raise ValueError("The images must have the same shape for MS-SSIM comparison.")

# Normalize the images to the range [0, 1]
image1 = image1 / image1.max()
image2 = image2 / image2.max()

# Calculate MS-SSIM
ms_ssim_value = ms_ssim(image1.unsqueeze(0), image2.unsqueeze(0))

# Print the result
print(f"MS-SSIM between the images: {ms_ssim_value.item():.4f}")
