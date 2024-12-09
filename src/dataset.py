import numpy as np
import config
import os
import tifffile
from torch.utils.data import Dataset, DataLoader
import volumentations.volumentations as v

import cv2

# Define a function to downsample using OpenCV
def downsample_image_opencv(image, target_size=(120, 120)):
    return cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_AREA)



class Dataset(Dataset):
    def __init__(self, root_dir_x, root_dir_y):
        self.root_dir_x = root_dir_x
        self.root_dir_y = root_dir_y
        self.list_files_x = os.listdir(self.root_dir_x)
        self.list_files_y = os.listdir(self.root_dir_y)
        if len(self.list_files_x) != len(self.list_files_y):
            print("X file list and Y file list are different sizes.")

    def __len__(self):
        return len(self.list_files_x)

    def __getitem__(self, index):
        img_file_x = self.list_files_x[index]
        img_path_x = os.path.join(self.root_dir_x, img_file_x)
        image_x = np.array(tifffile.imread(img_path_x))

        img_file_y = self.list_files_y[index]
        img_path_y = os.path.join(self.root_dir_y, img_file_y)
        image_y = np.array(tifffile.imread(img_path_y))

        #crop_tl = ((1128, 1788), (505, 1165))  # Crop size: (1291 - 291, 1610 - 610) = (1000, 1000)
        #crop_cb = ((132, 252), (90, 210))  # Crop size: (248 - 128, 122 - 2) = (120, 120)

        if config.RAW_TL_IMAGE == True and image_x.shape[0] == 4:
            # recreate not demosaiced image out of the mosaic
            input_image = np.empty(shape=(1, *image_x[0].shape), dtype=np.float32)
            input_image[0, 0::2, 0::2] = image_x[0, 0::2, 0::2] # top left, 0 deg
            input_image[0, 0::2, 1::2] = image_x[1, 0::2, 1::2] # top right, 45 deg
            input_image[0, 1::2, 0::2] = image_x[3, 1::2, 0::2] # bottom left, 135 deg or -45 deg
            input_image[0, 1::2, 1::2] = image_x[2, 1::2, 1::2] # bottom right, 90 deg
        elif config.RAW_TL_IMAGE == True and image_x.shape[0] == 5:
            input_image = np.array([image_x[4].astype('float32')])
        else:
            # Take polarization channel
            input_image = np.array([image_x[0].astype('float32')])
            # Crop to 660x660 and match FOV
            #input_image = input_image[:, crop_tl[1][0]:crop_tl[1][1], crop_tl[0][0]:crop_tl[0][1]]
            #input_image = downsample_image_opencv(input_image[0], target_size=(120, 120))
            #input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension -> (1, 120, 120)

            
        # norm
        min = np.min(input_image)
        max = np.max(input_image)
        input_image = (input_image - min) / (max - min + 1e-12)
        input_image = 2 * input_image - 1

        target_image = image_y.astype('float32')
        # Cropping to remove wavelengths > 800nm
        target_image = target_image[:93, :, :]
        #target_image = target_image[:93, crop_cb[1][0]:crop_cb[1][1], crop_cb[0][0]:crop_cb[0][1]]
        # Resizing CB image if needed/wanted
        if config.SHAPE_Y != target_image.shape:
            downsample = v.Compose([v.Resize(config.SHAPE_Y, always_apply=True)])
            target_image = downsample(image=target_image)["image"]
        # norm
        min = np.min(target_image)
        max = np.max(target_image)
        target_image = (target_image - min) / (max - min + 1e-12)
        target_image = 2 * target_image - 1

        # augmentations

        #augmentations = config.both_transform(image=input_image, image0=target_image)
        #input_image = augmentations["image"]
        #target_image = augmentations["image0"]

        #input_image = config.transform_only_input(image=input_image)["image"]
        #target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = Dataset(config.VAL_DIR_X, config.VAL_DIR_Y)
    loader = DataLoader(dataset, batch_size=1)
    for x, y in loader:
        print(x.shape)
        print(y.shape)
        # plt.imshow(x[0, 0, 400:450, 400:450])
        # plt.colorbar()
        # plt.show()
        break

