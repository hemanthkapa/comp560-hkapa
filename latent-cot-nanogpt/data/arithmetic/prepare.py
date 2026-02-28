"""
Data generator for arithmetic (addition) tasks.
Generates character-level addition problems: e.g., "123+45=168"
Used for testing latent Chain-of-Thought reasoning.
"""
import os
import pickle
import random
import numpy as np

MIN_DIGITS = 1
MAX_DIGITS = 2

target_length = 1_200_000
target_length_val = 200_000

def generate_addition_problem():
    """Generate a random addition problem and return as string."""
    d1 = random.randint(10 ** (MIN_DIGITS - 1), 10 ** MAX_DIGITS - 1)
    d2 = random.randint(10 ** (MIN_DIGITS - 1), 10 ** MAX_DIGITS - 1)
    result = d1 + d2
    return f"{d1}+{d2}={result}\n"

lines = []
total_length = 0
while total_length < target_length:
    line = generate_addition_problem()
    lines.append(line)
    total_length += len(line)

val_lines = []
total_length_val = 0
while total_length_val < target_length_val:
    line = generate_addition_problem()
    val_lines.append(line)
    total_length_val += len(line)

print("First 20 lines of training data:")
for i in range(20):
    print(lines[i].strip())

print("\nFirst 10 lines of validation data:")
for i in range(10):
    print(val_lines[i].strip())

data = ''.join(lines)
val_data = ''.join(val_lines)
print(f"\nlength of training dataset in characters: {len(data):,}")
print(f"length of validation dataset in characters: {len(val_data):,}")

chars = sorted(list(set(data + val_data)))
vocab_size = len(chars)
print(f"all unique characters: |{'|'.join(map(repr, chars))}|")
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

train_ids = encode(data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

data_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("\nData generation complete!")
print(f"Files saved to: {data_dir}")
