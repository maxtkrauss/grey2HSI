import torch
import torch.nn as nn
import torch.nn.functional as F
import config as c
import numpy as np
from torchsummary import summary

# Patch Size = 46x46

class Discriminator(nn.Module):
    def __init__(self, in_channels_x=1, in_channels_y=1, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

        self.sqrt = int(np.sqrt(c.NEAR_SQUARE))

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels_x + in_channels_y, features[0], kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], kernel_size=4, stride=2, padding=0),
            nn.InstanceNorm2d(features[1]),
            nn.LeakyReLU(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], kernel_size=4, stride=2, padding=0),
            nn.InstanceNorm2d(features[2]),
            nn.LeakyReLU(0.2)
        )
        
        self.layer4 = nn.Sequential( 
            nn.Conv2d(features[2], features[3], kernel_size=2, stride=1, padding=0),
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(0.2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(features[3], features[3], kernel_size=2, stride=1, padding=0),
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(0.2)
        )
        
        self.final = nn.Conv2d(features[3], 1, kernel_size=2)

    def forward(self, x, y):
            # Reshuffle y into spatial diffractive pattern
            diff = c.NEAR_SQUARE - c.SHAPE_Y[0]
            y = nn.functional.pad(y, (0, 0, 0, 0, 0, diff), mode='replicate')  # Pad if necessary
            y = nn.functional.pixel_shuffle(y, self.sqrt)  # Reshape y into spatial dimensions

            # Align the size of x and y using interpolation
            if y.shape[-1] < x.shape[-1]:  # If y is smaller, upsample y to match x
                y = nn.functional.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
            elif y.shape[-1] > x.shape[-1]:  # If x is smaller, upsample x to match y
                x = nn.functional.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate along the channel dimension
            x = torch.cat([x, y], dim=1)  # Combine x and y along channels

            # Pass through layers
            x = self.initial(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            patch_out = torch.sigmoid(self.final(x))  # Final patch prediction

            return patch_out
    
def test():
    # Example input
    x = torch.randn((1, 1, 660, 660))  # Source image
    y = torch.randn((1, 93, 120, 120))  # Target image
    model = Discriminator(in_channels_x=1, in_channels_y=1)
    pred = model(x, y)
    print("Shape of pred: ", pred.shape)

if __name__ == "__main__":
    test()
