import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):

    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (1, 320)

        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        #(1, 1280)
        return x
    
class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        

class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        #(batch_size, features, height, width) -> (batch_size, features, ht * 2, width *2)
        x = F.interpolate(x, scale_factor=2, mode = "nearest")
        return self.conv(x)

class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            #(batch_size, 4, height/8, width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            #(batch_size, 320, height/8, width/8) -> #(batch_size, 320, height/16, width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            #(batch_size, 640, height/16, width/16) -> #(batch_size, 640, height/32, width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            #(batch_size, 1280, height/32, width/32) -> #(batch_size, 1280, height/64, width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (batch_size, 1280, height/64, width/64) -> (batch_size, 1280, height/64, width/64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            #(batch_size, 2560, ht/64, width/64) -> (batch_size, 1280, ht/64, width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), #input is double as we factor in skip connection as well as output of bottleneck

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),


            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),


            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
        

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),


            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),


            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        #x: (batch_size, 320, ht/8, width/8)

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        #(batch_size, 4, ht/8, width/8)
        return x


class Diffusion(nn.Module):

    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        #latent: (batch_size, 4, height/8, width/8)
        #context: (batch_size, seq_len, dim)
        #time: (1, 320)

        #(1,320) -> (1, 1280)
        time = self.time_embedding(time)

        #(batch, 4, height/8, width/8) -> (batch, 320, height/8, width/8)
        output = self.unet(latent, context, time)

        #(batch, 320, height/8, width/8) -> (batch, 4, height/8, width/8) 
        output = self.final(output)

        # (batch, 4, height/8, width/8)
        return output
    
