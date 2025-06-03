import torch
from torch import nn 
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential): 
    # Eredita da sequential -> è un modello sequenziale

    def __init__(self):
        super().__init__(
            # (batch_size, Channel, h, w) -> (batch_size, 128, h, w)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (batch_size, 128, h, w) -> (batch_size, 128, h, w)
            VAE_ResidualBlock(128,128),
            # (batch_size, 128, h, w) -> (batch_size, 128, h/2, w/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2,padding=0),
            # (batch_size, 128, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(128,256),
            #  (batch_size, 256, h/2, w/2) -> (batch_size, 256, h/2, w/2)
            VAE_ResidualBlock(256,256),
            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h/4, w/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2,padding=0),
            # (batch_size, 256, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(256,512),
            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/4, w/4)
            VAE_ResidualBlock(512,512),
            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/8, w/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2,padding=0),
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512,512),
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512,512),
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512,512),
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_AttentionBlock(512),
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            VAE_ResidualBlock(512,512),
            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            nn.GroupNorm(32, 512),
            nn.SiLU(), #like ReLU 
            # (batch_size, 512, h/8, w/8) -> (batch_size, 8, h/8, w/8) bottleneck
            nn.Conv2d(512, 8, kernel_size=3, stride=2,padding=1),
            # (batch_size, 8, h/8, w/8) -> (batch_size, 8, h/8, w/8) 
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        ) # costruttore classe padre
        
    def forward(self, x: torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        """
        Processa l'immagine attraverso l'encoder.

        Args:
            x (torch.Tensor): immagine da codificare (batch_size, channel, h, w).
            noise (torch.Tensor): rumore con le dimensioni dell'output dell'encoder (batch_size, out_channels, h/8, w/8).

        Returns:
            torch.Tensor: un campione dello spazio latente (batch_size, out_channels, h/8, w/8)
        """

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                x = F.pad(x, (0,1,0,1)) # asymmetrical padding per alcuni strati
            x = module(x)
        
        # divide in due tensori lungo la dimensione 1:
        # (batch_size, 8, h/8, w/8) -> (batch_size, 4, h/8, w/8), (batch_size, 4, h/8, w/8)
        mean, log_variance = torch.chunck(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # per campionare, partiamo da un campione Z della distribuzione N(0,1) e trasformiamolo in un campione X della distribuzione N(mean, variance)
        # dalla teoria X = mean+stdev * Z

        x = mean + stdev * noise

        # scaling dell'output per una costante misteriosa
        x *= 0.18215

        return x