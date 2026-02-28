# Phase 2: LoopedGPT on arithmetic addition
# 2 layers looped 4x = 8 effective layers, ~2M params

out_dir = 'out/arithmetic_looped'
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

# looped model settings
model_type = 'looped'
n_layer = 2
n_head = 4
n_embd = 128
n_loop = 4
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