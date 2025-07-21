import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from datasets import load_dataset

from preprocessing import preprocess
from tokenizer import CodeTokenizer


INFERENCE_MODE = os.getenv('INFERENCE_MODE', '0') != '0'


# 0.6 billion parameters
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,           # Vocabulary size
    "context_length": 40_960,        # Context length that was used to train the model
    "emb_dim": 1024,                 # Embedding dimension
    "n_heads": 16,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
    "head_dim": 128,                 # Size of the heads in GQA
    "qk_norm": True,                 # Whether to normalize queries and values in GQA
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
}

# 1.7 billion parameters
QWEN3_CONFIG_1_7B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2048,                 # 2x larger than above
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 6144,              # 2x larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 4 billion parameters
QWEN3_CONFIG_4B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2560,                 # 25% larger than above
    "n_heads": 32,                   # 2x larger than above
    "n_layers": 36,                  # 29% larger than above
    "hidden_dim": 9728,              # ~3x larger than above
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 8 billion parameters
QWEN3_CONFIG_8B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 4096,                 # 60% larger than above
    "n_heads": 32,
    "n_layers": 36,                  # 26% larger than above
    "hidden_dim": 12288,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 14 billion parameters
QWEN3_CONFIG_14B = {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,                 # 25% larger than above
        "n_heads": 40,                   # 25% larger than above
        "n_layers": 40,                  # 11% larger than above
        "hidden_dim": 17408,             # 42% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
}

QWEN3_CONFIG_32B = {
        "vocab_size": 151_936,
        "context_length": 40_960,
        "emb_dim": 5120,
        "n_heads": 64,                   # 60% larger than above
        "n_layers": 64,                  # 60% larger than above
        "hidden_dim": 25600,             # 47% larger than above
        "head_dim": 128,
        "qk_norm": True,
        "n_kv_groups": 8,
        "rope_base": 1_000_000.0,
        "dtype": torch.bfloat16,
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # ModuleList since Sequential can only accept one input, and we need `x, mask, cos, sin`
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Reusable utilities
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx):
        # Forward pass
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin,)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        # Apply projections
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Reshape
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand K and V to match number of heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


def data_generator(dataset, tokenizer, block_size):
    """
    A generator that yields token sequences of a specified block size from the dataset.
    """
    buffer = []
    for sample in iter(dataset):
        content = preprocess(sample['content'])
        if not content:
            continue

        # Add a separator token between files
        content = content + " <|endoftext|>"
        tokens = tokenizer.encode(content).ids
        buffer.extend(tokens)

        while len(buffer) >= block_size + 1:
            chunk = buffer[:block_size + 1]
            buffer = buffer[block_size + 1:]
            yield torch.tensor(chunk, dtype=torch.long)


def get_batch(generator, batch_size, device):
    """
    Creates a batch of data from the data generator.
    """
    x_list, y_list = [], []
    for _ in range(batch_size):
        try:
            chunk = next(generator)
            x_list.append(chunk[:-1])
            y_list.append(chunk[1:])
        except StopIteration:
            # Not enough data for a full batch, can happen at the end
            break
    
    if not x_list:
        return None, None
        
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data_gen_factory, loss_fn, eval_iters, batch_size, device):
    out = {}
    model.eval()
    
    val_generator = data_gen_factory('val')
    
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_generator, batch_size, device)
        if X is None:
            losses = losses[:k]
            break
        logits = model(X)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = Y.view(B * T)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    
    out['val'] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def generate(model, tokenizer, start_string, max_new_tokens, device):
    """
    Performs inference to generate new text, safely preserving the model's training state.
    """
    # Store original training mode
    is_training = model.training
    model.eval()
    
    start_ids = tokenizer.encode(start_string).ids
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    block_size = model.cfg["context_length"]
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds model's block size
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        
        logits = model(x_cond)
        logits = logits[:, -1, :] # Get logits for the last token
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        
        # Stop if end of text token is generated
        if next_id.item() == tokenizer.eos_token_id:
            break
            
        x = torch.cat((x, next_id), dim=1)

    generated_text = tokenizer.decode(x[0].tolist())
    
    # Restore original training mode
    if is_training:
        model.train()
    
    breakpoint()
        
    return generated_text


def main():
    # Hyperparameters
    BATCH_SIZE = 8
    BLOCK_SIZE = 256
    LEARNING_RATE = 1e-4
    TRAINING_STEPS = 50000
    EVAL_INTERVAL = 500
    EVAL_ITERS = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BEST_MODEL_PATH = 'python_model.pth'

    print("Initializing tokenizer...")
    # pycharm people used a special tokenizer for python code
    tokenizer = CodeTokenizer().load('./tokenizer_training_data/custom_code_tokenizer.json')

    def data_gen_factory(split):
        if split == 'train':
            dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
        else:
            # For validation, we take a small, non-streamed part of the dataset to have consistent validation loss.
            # Here we are just re-using the start of the training set for simplicity. 
            # In a real scenario, you'd use a dedicated validation split.
            dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True).take(1000)
        return data_generator(dataset, tokenizer, BLOCK_SIZE)

    train_generator = data_gen_factory('train')

    model_config = QWEN_CONFIG_06_B
    model_config["vocab_size"] = tokenizer.get_vocab_size()
    model_config["context_length"] = BLOCK_SIZE
    
    model = Qwen3Model(model_config).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Using torch.compile for a speed-up
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Could not compile model: {e}. Running un-compiled.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    WARMUP_STEPS = round(TRAINING_STEPS * 0.1)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_STEPS - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.1)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_STEPS])

    print(f"\nStarting training on {DEVICE}...")
    best_val_loss = float('inf')

    pbar = tqdm(range(TRAINING_STEPS), desc='Training')
    for step in pbar:
        if step % EVAL_INTERVAL == 0 or step == TRAINING_STEPS - 1:
            losses = estimate_loss(model, data_gen_factory, loss_fn, EVAL_ITERS, BATCH_SIZE, DEVICE)
            val_loss = losses.get('val', float('inf'))
            print(f'\n\nStep {step}: validation loss {val_loss:.4f}')

            print(f"--- Generating sample text at step {step} ---")
            start_prompt = "import torch\n\ndef forward(self, x):"
            generated_text = generate(
                model, 
                tokenizer, 
                start_string=start_prompt, 
                max_new_tokens=100, 
                device=DEVICE
            )
            print(generated_text)
            print("--- End of sample ---\n")

            if val_loss < best_val_loss:
                # state_dict() on compiled model needs to be handled on the original model
                model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
                torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)
                print(f'New best val loss {best_val_loss:.4f} -> {val_loss:.4f}. Saving new model to {BEST_MODEL_PATH}\n')
                best_val_loss = val_loss

        xb, yb = get_batch(train_generator, BATCH_SIZE, DEVICE)
        if xb is None:
            print("Data generator exhausted. Ending training.")
            break

        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = yb.view(B * T)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_description(f"Training (loss: {loss.item():.4f}, lr: {current_lr:.6f})")

    print('Training finished.')

    if os.path.exists(BEST_MODEL_PATH):
        final_model_config = QWEN_CONFIG_06_B
        final_model_config["vocab_size"] = tokenizer.vocab_size
        final_model_config["context_length"] = BLOCK_SIZE
        model_to_load = Qwen3Model(final_model_config).to(DEVICE)
        model_to_load.load_state_dict(torch.load(BEST_MODEL_PATH))
        print("Loaded best model weights.")
    else:
        print("No best model found, using the final model for generation.")
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model


    print("\nGenerating final Python code sample...")
    start_prompt = "import torch\n\ndef forward(self, x):"
    generated_text = generate(model_to_load, tokenizer, start_string=start_prompt, max_new_tokens=150, device=DEVICE)
    print("\n--- FINAL GENERATED CODE ---")
    print(generated_text)

if __name__ == '__main__':
    raise SystemExit(main())
