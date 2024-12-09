import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.transforms.functional as tv_func
from src.utils import print_info
import time
import config as c

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False, kernel=2, stride=2, pad=1, output_padding=0):
        super(Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, pad, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, pad, output_padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(c.DROPOUT)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 2, 2, 0, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky", use_dropout=False)
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky", use_dropout=False)
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky", use_dropout=False)

        self.bottleneck = nn.Sequential(nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU())

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True, kernel=2, stride=2, pad=0, output_padding=1)
        self.up2 = Block(features * 8 + features * 8, features * 8, down=False, act="relu", use_dropout=True, kernel=4, stride=2, pad=1)
        self.up3 = Block(features * 8 + features * 8, features * 8, down=False, act="relu", use_dropout=True, kernel=2, stride=2, pad=0)
        self.up4 = Block(features * 8 + features * 8, features * 4, down=False, act="relu", use_dropout=False, kernel=4, stride=1, pad=0)
        self.up5 = Block(features * 4 + features * 4, features * 4, down=False, act="relu", use_dropout=False, kernel=4, stride=1, pad=0)
        self.up6 = Block(features * 4 + features * 2, features * 2, down=False, act="relu", use_dropout=False, kernel=4, stride=2, pad=1, output_padding = 1)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2 + features, out_channels, kernel_size=4, stride=1, padding=0),
            nn.Tanh(),
        )
    
    def forward(self, x):
        # Input: 1x900x900
       # print("Initial shape:", x.shape)
        d1 = self.initial_down(x) # 64x450x450
        #print("Down 1:", d1.shape)
        d2 = self.down1(d1) # 128x225x225
        #print("Down 2:", d2.shape)
        d3 = self.down2(d2) # 256x112x112
        #print("Down 3:",d3.shape)
        d4 = self.down3(d3) # 512x56x56
        #print("Down 4:", d4.shape)
        d5 = self.down4(d4) # 512x28x28
        #print("Down 5:", d5.shape)
        d6 = self.down5(d5) # 512x14x14
        #print("Down 6:", d6.shape)
        bottleneck = self.bottleneck(d6) # 512x7x7
        #print("Bottleneck:", bottleneck.shape)
        # print("bn:", bottleneck.shape)
        u1 = self.up1(bottleneck) # 512x13x13
        #print("Up 1:", u1.shape)
        # print("u1:", u1.shape, "d6:", d6.shape)
        
        d6_ip = func.interpolate(d6, u1.shape[2:], mode='bilinear')
        u2 = self.up2(torch.cat([u1, d6_ip], 1)) # 512x26x26
       #print("Up 2:", u2.shape)
        # print("u2:", u2.shape, "d5:", d5.shape)
        d5_ip = func.interpolate(d5, u2.shape[2:], mode='bilinear')
        u3 = self.up3(torch.cat([u2, d5_ip], 1)) # 512x52x52
        #print("Up 3:", u3.shape)
        # print("u3:", u3.shape, "d4:", d4.shape)
        d4_ip = func.interpolate(d4, u3.shape[2:], mode='bilinear')
        u4 = self.up4(torch.cat([u3, d4_ip], 1)) # 256x52x52
        #print("Up 4:", u4.shape)
        # print("u4:", u4.shape, "d3:", d3.shape)
        d3_ip = func.interpolate(d3, u4.shape[2:], mode='bilinear')
        u5 = self.up5(torch.cat([u4, d3_ip], 1)) # 256x104x104
        #print("Up 5:", u5.shape)
        # print("u5:", u5.shape, "d2:", d2.shape)
        d2_ip = func.interpolate(d2, u5.shape[2:], mode='bilinear')
        u6 = self.up6(torch.cat([u5, d2_ip], 1)) # 128x104x104
        #print("Up 6:", u6.shape)
        # print("u6:", u6.shape, "d1:", d1.shape)
        d1_ip = func.interpolate(d1, u6.shape[2:], mode='bilinear')
        u7 = self.final_up(torch.cat([u6, d1_ip], 1)) # 106x104x104
        #print("Final shape:", u7.shape)
        return u7

def test():
    start=time.time()
    x = torch.randn((1, 1, 660, 660))
    model = Generator(in_channels=1, out_channels=93, features=64)
    preds = model(x)
    end=time.time()
    print("\nShape of prediction:\n", preds.shape)
    print_info(preds, "Preds")
    print(f"Time (s): {(end-start):.2f}")


if __name__ == "__main__":
    test()
