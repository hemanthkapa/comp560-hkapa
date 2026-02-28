"""
Evaluate a trained model on arithmetic addition with exact-match accuracy.
Generates random addition problems, feeds the prompt up to '=', and checks
whether the model produces the correct answer.

Usage:
    python evaluate.py config/arithmetic.py --out_dir=out/arithmetic_baseline
    python evaluate.py config/arithmetic_looped.py --out_dir=out/arithmetic_looped
"""

import os
import pickle
import random
from collections import defaultdict
from contextlib import nullcontext

import torch
from model import GPTConfig, GPT, LoopedGPT

# -----------------------------------------------------------------------------
# defaults (overridable via config file / CLI)
out_dir = 'out/arithmetic_baseline'
num_problems = 200
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype = 'float32'
dataset = 'arithmetic'
model_type = 'gpt'
n_loop = 1
compile = False
config_file = os.environ.get("NANOGPT_CONFIG", "configurator.py")
exec(open(config_file).read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
random.seed(seed)

device_type = 'cuda' if 'cuda' in device else 'mps' if 'mps' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type in ('cpu', 'mps') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load model from checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
ModelClass = LoopedGPT if model_type == 'looped' else GPT
model = ModelClass(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)

# load vocab from meta.pkl
meta_path = os.path.join('data', dataset, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

newline_token = stoi['\n']


def generate_answer(prompt_str, max_tokens=12):
    """Feed prompt to model and generate until newline or max_tokens."""
    prompt_ids = encode(prompt_str)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]
    with torch.no_grad():
        with ctx:
            for _ in range(max_tokens):
                x_cond = x if x.size(1) <= gptconf.block_size else x[:, -gptconf.block_size:]
                logits, _ = model(x_cond)
                logits = logits[:, -1, :]
                next_id = logits.argmax(dim=-1, keepdim=True)
                x = torch.cat([x, next_id], dim=1)
                if next_id.item() == newline_token:
                    break
    generated_ids = x[0, len(prompt_ids):].tolist()
    return decode(generated_ids).strip()


def max_digits(a, b):
    return max(len(str(a)), len(str(b)))


print(f"Evaluating {num_problems} problems from checkpoint: {ckpt_path}")
print(f"Device: {device}")
print(f"Model type: {model_type}\n")

results_by_digits = defaultdict(lambda: {'correct': 0, 'total': 0})
total_correct = 0

for i in range(num_problems):
    d1 = random.randint(1, 99)
    d2 = random.randint(1, 99)
    expected = d1 + d2
    prompt = f"{d1}+{d2}="

    generated = generate_answer(prompt)

    digit_bucket = max_digits(d1, d2)
    results_by_digits[digit_bucket]['total'] += 1

    try:
        predicted = int(generated)
    except ValueError:
        predicted = None

    correct = (predicted == expected)
    if correct:
        total_correct += 1
        results_by_digits[digit_bucket]['correct'] += 1

    if i < 20 or not correct:
        status = "OK" if correct else "WRONG"
        print(f"  {prompt}{expected}  |  model: {prompt}{generated}  [{status}]")

print(f"\n{'='*50}")
print(f"Overall: {total_correct}/{num_problems} = {100*total_correct/num_problems:.1f}%")
print(f"\nBreakdown by max operand digits:")
for d in sorted(results_by_digits.keys()):
    r = results_by_digits[d]
    pct = 100 * r['correct'] / r['total'] if r['total'] > 0 else 0
    print(f"  {d}-digit: {r['correct']}/{r['total']} = {pct:.1f}%")
