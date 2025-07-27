import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from datasets import load_dataset
import numpy as np

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


class StreamingTokenizedDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
    
    def __iter__(self):
        buffer = []

        for sample in self.dataset:
            content = preprocess(sample['content'])
            if not content:
                continue

            # Add a separator token between files
            content = content + " <|endoftext|>"
            tokens = self.tokenizer.encode(content).ids
            buffer.extend(tokens)

            # Yield complete blocks of tokens
            while len(buffer) >= self.block_size + 1:
                chunk = torch.tensor(buffer[:self.block_size + 1], dtype=torch.long)
                buffer = buffer[self.block_size + 1:]
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y


class FullLineGenerator:
    def __init__(self, model, tokenizer, device, beam_width=5, max_iterations=20, stop_factor_k=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.stop_factor_k = stop_factor_k

        # Get the vocabulary for token healing. Assumes tokenizer has a get_vocab() method.
        self.vocab = self.tokenizer.get_vocab().keys()
        # The blog post specifically mentions collecting hypotheses ending in newline.
        # Ensure your tokenizer includes '\n' or use an appropriate end-of-line token.
        self.newline_token_id = self.tokenizer.token_to_id("\n")
        if self.newline_token_id is None:
            raise ValueError("Tokenizer must have a newline '\\n' character in its vocabulary.")

    @torch.no_grad()
    def _token_healing(self, line_prefix):
        for i in range(len(line_prefix), -1, -1):
            suffix = line_prefix[i:]
            if self.tokenizer.token_to_id(suffix) is not None:
                continue

            prefix = line_prefix[:i]
            if not prefix:
                return "", suffix
            
            prefix_ids = self.tokenizer.encode(prefix).ids
            reconstruced_prefix = self.tokenizer.decode(prefix_ids)

            if reconstruced_prefix == prefix:
                return prefix, suffix
            
        return "", line_prefix

    @torch.no_grad()
    def generate(self, current_line):
        """
        Generates a line completion using a modified beam search.
        """
        is_training = self.model.training
        self.model.eval()

        # 1. TOKEN HEALING
        healed_context, prefix_to_filter = self._token_healing(current_line)
        
        context_ids = self.tokenizer.encode(healed_context).ids
        
        # Initialize beams: (log_probability, sequence_of_ids)
        active_beams = [(-0.0, context_ids)]
        terminated_hypotheses = []
        block_size = self.model.cfg["context_length"]

        # 2. DYNAMIC BEAM SEARCH LOOP
        for step in range(self.max_iterations):
            if not active_beams:
                break

            all_new_candidates = []
            for log_prob, sequence in active_beams:
                input_ids = sequence[-block_size:]
                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
                
                logits = self.model(input_tensor)[:, -1, :]
                next_token_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                top_k_log_probs, top_k_ids = torch.topk(next_token_log_probs, self.beam_width, dim=-1)

                for i in range(self.beam_width):
                    next_id = top_k_ids[0, i].item()
                    new_log_prob = log_prob + top_k_log_probs[0, i].item()
                    new_sequence = sequence + [next_id]
                    all_new_candidates.append((new_log_prob, new_sequence))

            # Filter candidates based on the "healed" prefix (only on the first step)
            if step == 0 and prefix_to_filter:
                filtered_candidates = []
                for log_prob, seq in all_new_candidates:
                    # Decode only the newly generated part for the check
                    generated_part = self.tokenizer.decode(seq[len(context_ids):])
                    if generated_part.startswith(prefix_to_filter):
                        filtered_candidates.append((log_prob, seq))
                all_new_candidates = filtered_candidates
            
            active_beams = []
            # Sort all potential new beams by their score
            all_new_candidates.sort(key=lambda x: x[0], reverse=True)
            
            for log_prob, sequence in all_new_candidates[:self.beam_width]:
                if sequence[-1] == self.newline_token_id:
                    terminated_hypotheses.append((log_prob, sequence))
                else:
                    active_beams.append((log_prob, sequence))
            
            # Dynamic stopping criteria
            if terminated_hypotheses:
                best_terminated_score, _ = max(terminated_hypotheses, key=lambda x: x[0])
                # Stop if all active beams are much worse than the best completed one
                if not active_beams or max(active_beams, key=lambda x: x[0])[0] < best_terminated_score - np.log(self.stop_factor_k):
                    break
            
        # Restore model's original training state if it was training
        if is_training:
            self.model.train()
        
        # If no suggestion ends in a newline, return nothing
        if not terminated_hypotheses:
            print('Info: Found no terminated hypotheses')
            return ""
        
        non_empty_hypotheses = []
        for log_prob, sequence in terminated_hypotheses:
            generated_text = self.tokenizer.decode(sequence[len(context_ids):]).strip()
            if generated_text:
                non_empty_hypotheses.append((log_prob, sequence))
        
        if non_empty_hypotheses:
            _, best_sequence = max(non_empty_hypotheses, key=lambda x: x[0])
        else:
            _, best_sequence = max(terminated_hypotheses, key=lambda x: x[0])

        full_suggestion_text = self.tokenizer.decode(best_sequence)

        # Return only the newly generated part of the text
        return full_suggestion_text[len(healed_context):]


@torch.no_grad()
def estimate_loss(model, val_loader, loss_fn, eval_iters, device):
    out = {}
    model.eval()
    
    losses = torch.zeros(eval_iters)
    val_iterator = iter(val_loader)

    for k in range(eval_iters):
        X, Y = next(val_iterator)
        X, Y = X.to(device), Y.to(device)
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

        top_k = 5
        temperature = 1.0
        logits = logits[:, -1, :] # Get logits for the last token
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(device), logits)

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        # probs = torch.softmax(logits, dim=-1)
        # next_id = torch.multinomial(probs, num_samples=1)
        
        # Stop if end of text token is generated
        if next_id.item() == tokenizer.eos_token_id:
            break
            
        x = torch.cat((x, next_id), dim=1)

    generated_text = tokenizer.decode(x[0].tolist())
    
    # Restore original training mode
    if is_training:
        model.train()
    
    return generated_text


def main():
    # Hyperparameters
    BATCH_SIZE = 3
    BLOCK_SIZE = 512
    LEARNING_RATE = 5e-5
    TRAINING_STEPS = 600_000
    EVAL_INTERVAL = 10_000
    EVAL_ITERS = 1_000
    VAL_SET_SIZE = 3_000
    NUM_WORKERS = min(max(1, os.cpu_count() // 2), 3)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BEST_MODEL_PATH = 'python_model.pth'

    print("Initializing tokenizer...")
    tokenizer = CodeTokenizer.load('./tokenizer_training_data/custom_code_tokenizer.json')

    data_stream = load_dataset('bigcode/the-stack', data_dir='data/python', split='train', streaming=True)
    raw_val_dataset = data_stream.take(VAL_SET_SIZE)
    raw_train_dataset = data_stream.skip(VAL_SET_SIZE)
    val_dataset = StreamingTokenizedDataset(raw_val_dataset, tokenizer, BLOCK_SIZE)
    train_dataset = StreamingTokenizedDataset(raw_train_dataset, tokenizer, BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    model_config = QWEN3_CONFIG_1_7B 
    model_config["vocab_size"] = tokenizer.get_vocab_size()
    model_config["context_length"] = BLOCK_SIZE
    
    model = Qwen3Model(model_config).to(DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    if os.path.exists(BEST_MODEL_PATH):
        print('Loading model')
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Model weights loaded successfully from {BEST_MODEL_PATH}")

    # Using torch.compile for a speed-up
    model = torch.compile(model, fullgraph=True)
    
    generator = FullLineGenerator(model=model, tokenizer=tokenizer, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    WARMUP_STEPS = round(TRAINING_STEPS * 0.1)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=TRAINING_STEPS - WARMUP_STEPS, eta_min=LEARNING_RATE * 0.1)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_STEPS])

    print(f"\nStarting training on {DEVICE}...")
    best_val_loss = float('inf')

    val_propmt = '''\
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
'''

    print('Starting training')
    pbar = tqdm(enumerate(train_loader), desc='Training')
    for step, (xb, yb) in pbar:
        if step >= TRAINING_STEPS:
            break
        if (step + 1) % EVAL_INTERVAL == 0 or step == TRAINING_STEPS - 1:
            losses = estimate_loss(model, val_loader, loss_fn, EVAL_ITERS, DEVICE)
            val_loss = losses.get('val', float('inf'))
            print(f'\n\nStep {step}: validation loss {val_loss:.4f}')

            print(f"--- Generating beam search line at step {step} ---")
            # suggestion = generator.generate(current_line=start_prompt)
            # print(start_prompt + suggestion)
            print('todo line complete disabled')
            print(f"--- Generating sample text at step {step} ---")
            generated_text = generate(
                model, 
                tokenizer, 
                start_string=val_propmt, 
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

        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
        if step % 50 == 0:
            pbar.set_description(f"Training (loss: {loss.item():.4f}, lr: {current_lr:.6f})")

    print('Training finished.')

    if os.path.exists(BEST_MODEL_PATH):
        final_model_config = QWEN3_CONFIG_1_7B
        final_model_config["vocab_size"] = tokenizer.vocab_size
        final_model_config["context_length"] = BLOCK_SIZE
        model_to_load = Qwen3Model(final_model_config).to(DEVICE)
        model_to_load.load_state_dict(torch.load(BEST_MODEL_PATH))
        print("Loaded best model weights.")
    else:
        print("No best model found, using the final model for generation.")
        model_to_load = model._orig_mod if hasattr(model, '_orig_mod') else model


    print("\nGenerating final Python code sample...")
    generated_text = generate(model_to_load, tokenizer, start_string=val_propmt, max_new_tokens=150, device=DEVICE)
    print("\n--- FINAL GENERATED CODE ---")
    print(generated_text)

if __name__ == '__main__':
    raise SystemExit(main())
