import torch

# Data Config
NEPTUNE_PROJECT = "NASA-HSI-MK/NASA-HSI"
TRAIN_DIR_X = "//scratch//general//nfs1//u1528328//img_dir//gg_filter_test_3//gg_filter_test_3//train//thorlabs"
TRAIN_DIR_Y = "//scratch//general//nfs1//u1528328//img_dir//gg_filter_test_3//gg_filter_test_3//train//cubert"

VAL_DIR_X = "//scratch//general//nfs1//u1528328//img_dir//gg_filter_test_3//gg_filter_test_3//val//thorlabs"
VAL_DIR_Y = "//scratch//general//nfs1//u1528328//img_dir//gg_filter_test_3//gg_filter_test_3//val//cubert"

MODEL_DIR = "//scratch//general//nfs1//u1528328//model_dir//MK_12-9_test"

TEST_DIR_X = VAL_DIR_X
TEST_DIR_Y = VAL_DIR_Y

SHAPE_X =  (1, 660, 660)  
SHAPE_Y =  (93, 120, 120)
NEAR_SQUARE = 121 # Nearest square number to the spectral dim of Y, change this when changing SHAPE_Y
RAW_TL_IMAGE = False

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATOR_MODEL = "MK_unet" # unet, unet2d, fp_unet, simple_fp_unet, outside_fp_unet, MK_unet, spectral_unet
LEARNING_RATE = 0.0002 # Starting learning rate
LR_GAMMA = 0.9992 # Gamma for exponential decay function
LR_START_DECAY = 750 # start exp decay after x steps
DROPOUT = 0.2 # Dropout used for the generator

BATCH_SIZE =  1
NUM_WORKERS = 2

# Generator loss function
LAMBDA_ADV = 0.5
LAMBDA_L1 = 100
LAMBDA_L2 = 0
LAMBDA_SAM = 25
LAMBDA_LFM = 0
LAMBDA_RASE = 0
LAMBDA_SSIM = 50
LAMBDA_MS_SSIM = 0
LAMBDA_MSE = 100  

# Training
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
LOG_IMAGES = True
# CHECKPOINT_DISC = "model/disc.pth.tar"
# CHECKPOINT_GEN = "model/gen.pth.tar"

#-------------------To Do---------------------
# Constant LR
# Reshuffle/mosiacing discriminator input
# Generator max pooling.... why?
