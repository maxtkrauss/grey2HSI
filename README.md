This project implements a Pix2Pix network to translate greyscale images into hyperspectral images (HSI). The network is based on a generator-discriminator architecture, where the generator learns to produce realistic HSI from greyscale inputs, and the discriminator evaluates their authenticity.

The network is trained on inputs of (1, 660, 660) greyscale and (93, 120, 120) HSI. After training, feed it a greyscale image to generate a HSI reconstruction. The network is still in its early stages of development, but it operates as intended, successfully generating HSI reconstructions from greyscale inputs. However, the quality of the reconstructions is not yet satisfactory for practical applications.

The implementation includes a customizable loss functions, data configurations, and model architecture options. The project uses PyTorch Lightning for modular training and Neptune for experiment tracking. 
