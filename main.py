import os
import json
import re
import urllib.request
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


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


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    # Model uses weight tying, hence we reuse the embedding layer weights here
    model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None,
                 apply_chat_template=True, add_generation_prompt=False, add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        if not tok_file.is_file() and repo_id:
            download_from_huggingface(
                repo_id=repo_id,
                filename=tok_file.name,
                local_dir=str(tok_file.parent),
            )
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token_id = self._special_to_id.get("<|endoftext|>")
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            text = self._wrap_chat(text)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
    else:
        print(f"Downloading {url} to {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)

    return dest_path


def download_from_huggingface_from_snapshots(repo_id, local_dir):
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file  # or your preferred loader

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    if os.path.exists(index_path):
        # Multi-shard model
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    elif os.path.exists(single_file_path):
        # Single-shard model
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        raise FileNotFoundError("No model.safetensors or model.safetensors.index.json found.")

    return weights_dict


def download_shakespeare_data(url, save_path):
    """Downloads the Tiny Shakespeare dataset."""
    if not os.path.exists(save_path):
        print("Downloading Tiny Shakespeare dataset...")
        response = requests.get(url)
        with open(save_path, 'w') as f:
            f.write(response.text)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")

def get_data_loader(text, batch_size, block_size, device):
    """Creates a simple data loader for the Shakespeare text."""
    # Create a character-level vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    return get_batch, vocab_size, decode

@torch.no_grad()
def estimate_loss(model, get_batch, loss_fn, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = Y.view(B * T)
            loss = loss_fn(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def generate(model, start_string, max_new_tokens, decode, device):
    """Performs inference to generate new text."""
    model.eval()
    # For a character-level model, we need to create the encoder here as well
    chars = sorted(list(set(open('shakespeare.txt', 'r').read())))
    stoi = {ch: i for i, ch in enumerate(chars)}
    
    start_ids = [stoi[c] for c in start_string]
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    block_size = model.cfg["context_length"]
    
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits = model(x_cond)
        logits = logits[:, -1, :] # Get logits for the last token
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_id), dim=1)

    return decode(x[0].tolist())


def main():
    # Hyperparameters
    BATCH_SIZE = 16
    BLOCK_SIZE = 32
    LEARNING_RATE = 1e-3
    TRAINING_STEPS = 100000
    EVAL_INTERVAL = 1000
    EVAL_ITERS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Download data
    DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    DATA_PATH = 'shakespeare.txt'
    download_shakespeare_data(DATA_URL, DATA_PATH)

    BEST_MODEL_PATH = 'model.pth'

    # Load data
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    get_batch, vocab_size, decode = get_data_loader(text, BATCH_SIZE, BLOCK_SIZE, DEVICE)

    # For this small dataset, we will override the config
    # to create a much smaller model.
    shakespeare_config = {
        "vocab_size": vocab_size,
        "context_length": BLOCK_SIZE,
        "emb_dim": 128,
        "n_heads": 4,
        "n_layers": 4,
        "hidden_dim": 256,
        "head_dim": 32,
        "qk_norm": False,
        "n_kv_groups": 2,
        "rope_base": 10000.0,
        "dtype": torch.bfloat16,
    }

    # Initialize model, optimizer, and loss
    model = Qwen3Model(shakespeare_config).to(DEVICE)
    model = torch.compile(model)
    # model = Qwen3Model(QWEN_CONFIG_06_B).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    WARMUP_STEPS = round(TRAINING_STEPS * 0.2)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=WARMUP_STEPS)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_STEPS - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_STEPS])

    # Train the model
    print("\nStarting training...")
    best_val_loss = float('inf')

    for step in tqdm(range(TRAINING_STEPS), desc='Training'):
        if step % EVAL_INTERVAL == 0 or step == TRAINING_STEPS - 1:
            losses = estimate_loss(model, get_batch, loss_fn, EVAL_ITERS)
            print(f'\nStep {step}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

            if losses['val'] < best_val_loss:
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f'New best val loss {best_val_loss:.4f} -> {losses["val"]:.4f}. Saving new model to {BEST_MODEL_PATH}')
                best_val_loss = losses['val']

        xb, yb = get_batch('train')

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
    
    print('Training finished.')

    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print("Loaded best model weights for generation.")
    else:
        print("No best model found, using the final model for generation.")

    # Perform inference
    print("\nTraining finished. Generating text...")
    start_prompt = "JULIET:\nO Romeo, Romeo! wherefore art thou"
    generated_text = generate(model, start_prompt, max_new_tokens=200, decode=decode, device=DEVICE)
    print("\n--- GENERATED TEXT ---")
    print(generated_text)

if __name__ == '__main__':
    raise SystemExit(main())
