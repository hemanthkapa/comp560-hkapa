# Phase 1 baseline: 4-layer standard GPT on arithmetic addition

out_dir = 'out/arithmetic_baseline'
eval_interval = 100
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = None
wandb_run_name = None

dataset = 'arithmetic'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 32

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 200

device = 'cpu'
dtype = 'float32'
compile = False
