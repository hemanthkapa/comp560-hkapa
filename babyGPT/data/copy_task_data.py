import torch

def get_batch(batch_size=32, block_size=16, vocab_size=10, device='cpu'):
    # Generate the pattern (half the block size)
    # The model will see this pattern and then must learn to repeat it.
    half_block = block_size // 2
    
    # Generate random integers. This is the "source" information.
    # Shape: (batch_size, half_block) e.g., [1, 5, 2]
    data = torch.randint(0, vocab_size, (batch_size, half_block))

    # Create the Copy Task
    # Concatenate the data with itself.
    data = torch.cat([data, data], dim=1)

    # Create Inputs (x) and Targets (y)
    # y is x shifted by one
    x = data[:, :-1]
    y = data[:, 1:]

    x, y = x.to(device), y.to(device)

    return x, y

if __name__ == '__main__':
    x, y = get_batch(batch_size=4, block_size=8, vocab_size=10)
    print("Input (x): ")
    print(x)
    print("Target (y): ")
    print(y)
    print(f"Shapes: x={x.shape}, y={y.shape}")
