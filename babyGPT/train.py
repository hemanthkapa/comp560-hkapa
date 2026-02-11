import torch
import torch.nn as nn
from model.transformer import GPT
from data.copy_task_data import get_batch
import wandb

# 1. Hyperparameters
batch_size = 32
block_size = 64
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model
n_layer = 4
n_head = 4
n_embd = 128
vocab_size = 10 

# Weights & Biases
wandb_log = True
wandb_project = 'babyGPT-copy'
wandb_run_name = 'copy-v1'

if wandb_log:
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "batch_size": batch_size,
            "block_size": block_size,
            "learning_rate": learning_rate,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "device": device,
        }
    )

# 2. Instantiate Model
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size, device=device)
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6}M")

# 3. Create Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 4. Training Loop
print("Starting training...")
for iter in range(max_iters):
    # Sample a batch of data
    xb, yb = get_batch(batch_size, block_size, vocab_size, device)

    # Evaluate the loss
    logits, loss = model(xb, yb)
    
    # Optimize
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % eval_interval == 0:
        print(f"step {iter}: loss {loss.item():.4f}")
        if wandb_log:
            wandb.log({"iter": iter, "train/loss": loss.item()})

print(f"Final training loss: {loss.item():.4f}")
wandb.finish()

print("Generative Demo:")
# Create a random context of half the block size (32 tokens)
half_block = block_size // 2
context = torch.randint(0, vocab_size, (1, half_block), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=12)[0].tolist()
print(generated)
