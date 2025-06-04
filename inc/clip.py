import torch
from torch import nn
from torch.nn import functional as F
from VAE.attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab:int, n_embd:int, n_tokens:int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embd) #oggetto che implementa il metodo __call__() quindi si comporta come una funzione
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))

    def forward(self, tokens):
        # (Batch_size, Seq_len) -> (Batch_size, seq_len, dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return 0

class CLIPLayer(nn.Module):
    def __init__(self, n_head:int, n_embd:int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, 4*n_embd)
        self.linear2 = nn.Linear(4*n_embd, n_embd)

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # (Batch_size, seq_len, dim)

        residue = x

        # Self attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residue

        # feedforward layer
        residue = x
        
        x = self.layernorm_2(x)
        x = self.linear1(x)

        x = x*torch.sigmoid(1.702*x) # QuickGeLU activation

        x+=residue

        return x
    


class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module(
            CLIPLayer(12, 768) for i in range(12)
        ) 
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens:torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_size, Seq_len) -> (Batch_size, seq_len, dim)
        state = self.embedding(tokens)


        for layer in self.layers:
            state = self.layers(state)
        
        # (Batch_size, seq_len, dim) -> (Batch_size, seq_len, dim)
        output = self.layernorm(state)
        return output

