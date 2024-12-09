This project implements a Pix2Pix network to translate greyscale images into hyperspectral images (HSI). The network is based on a generator-discriminator architecture, where the generator learns to produce realistic HSI from greyscale inputs, and the discriminator evaluates their authenticity.

The implementation includes customizations for spectral loss functions, data configurations, and model architecture options. The project leverages PyTorch Lightning for modular training and Neptune for experiment tracking.
