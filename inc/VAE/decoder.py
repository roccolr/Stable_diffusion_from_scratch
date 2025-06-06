import torch 
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels:int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        """
        Processa l'input attraverso il blocco Attention.

        Args:
            x (torch.Tensor): input (batch_size, features, h, w).

        Returns:
            torch.Tensor: tensore delle stesse dimensione dell'input da passare al livello successivo.
        """

        residue = x
        n, c, h, w = x.shape

        # (batch_size, features, h, w) -> (batch_size, features, h * w)
        x = x.view(n, c, h*w)

        # (batch_size, features, h * w) -> (batch_size, h, w, features)
        x = x.transpose(-1, -2)

        x = self.attention(x)

        # (batch_size, h, w, features) -> (batch_size, features, h * w) 
        x = x.transpose(-1, -2)

        # (batch_size, features, h * w) -> (batch_size, features, h, w)   
        x = x.view(n, c, h*w)

        x+=residue 
        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 3, padding=0)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Processa l'input attraverso il blocco Residual.

        Args:
            x (torch.Tensor): input (batch_size, in_channel, h, w).

        Returns:
            torch.Tensor: tensore delle stesse dimensione dell'input da passare al livello successivo.
        """

        residue = x 
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + residue

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4 , kernel_size=1, padding=0),
            nn.Conv2d(4, 512 , kernel_size=3, padding=1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            # (Batch_size, 512, h/8, w/8) -> (Batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512,512),

            # (Batch_size, 512, h/8, w/8) -> (Batch_size, 512, h/4, w/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512 , kernel_size=3, padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            # (Batch_size, 512, h/4, w/4) -> (Batch_size, 512, h/2, w/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512 , kernel_size=3, padding=1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            # (Batch_size, 256, h/2, w/2) -> (Batch_size, 256, h, w)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256 , kernel_size=3, padding=1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            
            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # (Batch_size, 128, h, w) -> (Batch_size, 3, h, w) i 3 canali RGB
            nn.Conv2d(128, 3 , kernel_size=3, padding=1)
        )

    def forward(self, x:torch.Tensor)-> torch.Tensor:

        """
        Processa il tensore latente attraverso il decoder.

        Args:
            x (torch.Tensor): tensore latente da decodificare (batch_size, 4, h/8, w/8).

        Returns:
            torch.Tensor: immagine ricostruita (Batch_size, 3, h, w)
        """

        x /= 0.18215
        for module in self:
            x = module(x)
        
        return x
    

