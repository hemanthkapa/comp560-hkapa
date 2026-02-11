import torch
import torch.nn as nn
from torch.nn import functional as F
from model.modules import Block

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, device='cpu'):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head, block_size=block_size)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd) 
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Token Embeddings (what is the token?)
        tok_emb = self.token_embedding_table(idx)
        
        # 2. Position Embeddings (where is the token?)
        pos_idx = torch.arange(T, device=self.device)
        pos_emb = self.position_embedding_table(pos_idx)
        
        # 3. Combine
        x = tok_emb + pos_emb # (B, T, C)
        
        # 4. Run through Blocks
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)   # (B, T, C)
        
        # 5. Get Logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            # Reshape for Loss Calculation
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.position_embedding_table.num_embeddings else idx[:, -self.position_embedding_table.num_embeddings:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

