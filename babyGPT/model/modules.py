import torch 
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size, n_embd, block_size):
        # head_size: the dimension of the key, query, and value vectors (taking n_embd and splitting across heads)
        # n_embd: the dimension of the input embeddings (how many features)
        # block_size: the maximum length of the input sequence (How far back can we look?)

        super().__init__()
        # The projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # The Mask
        #'tril' stands for lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, channels)
        B, T, C = x.shape 
        # B = batch size, T = time-step C = channels
        # batch size = how many independent sequences we are processing at once
        # time-step = how many tokens in the sequence
        # channels = the dimension of the input embeddings (how many features)

        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # compute attention scores
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # Dot product and scaling
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Masking
        weights = F.softmax(weights, dim=-1) # Softmax
        out = weights @ v # Weighted sum
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size):
        super().__init__()

        #create a list of num_heads independent attention heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size) for _ in range(num_heads)])
        #Concatenate the outputs of the heads and project them to the original dimension
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        # Run each head independently and concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        #Project the concatenated outputs to the original dimension
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Expand standard dimension by 4 (standard Transformer practice)
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Project back down
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)