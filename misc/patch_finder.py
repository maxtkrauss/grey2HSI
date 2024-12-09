import torch
import torch.nn as nn

# Define the custom Discriminator model
class Discriminator(nn.Module):
    def __init__(self, in_channels_x=1, in_channels_y=93, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()

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
            nn.BatchNorm2d(features[3]),
            nn.LeakyReLU(0.2)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(features[3], features[3], kernel_size=2, stride=1, padding=0),
            nn.InstanceNorm2d(features[3]),
            nn.LeakyReLU(0.2)
        )
        
        self.final = nn.Conv2d(features[3], 1, kernel_size=2)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.final(x)

# Function to calculate the patch size
def calculate_patch_size(model, input_shape=(107, 256, 256)):
    # Initialize parameters
    receptive_field = 1
    stride = 1

    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Conv2d):
                    kernel_size = sub_layer.kernel_size[0]
                    layer_stride = sub_layer.stride[0]
                    

                    # Calculate receptive field expansion
                    receptive_field += (kernel_size - 1) * stride
                    stride *= layer_stride  # Update effective stride

        elif isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]
            layer_stride = layer.stride[0]
            
            # Calculate receptive field expansion
            receptive_field += (kernel_size - 1) * stride
            stride *= layer_stride  # Update effective stride

    return receptive_field

# Instantiate the model
discriminator_model = Discriminator()

# Calculate the patch size
patch_size = calculate_patch_size(discriminator_model, input_shape=(2, 1200, 1200))
print(f"The patch size (receptive field) of the Discriminator is: {patch_size}x{patch_size}")
