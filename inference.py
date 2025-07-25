import argparse
import torch
import sys

from main import Qwen3Model, QWEN3_CONFIG_1_7B
from tokenizer import CodeTokenizer


@torch.no_grad()
def generate_sampling(model, tokenizer, prompt, num_lines, max_new_tokens, device, top_k : int = 10, temperature: float = 1.0):
    """Generates text using multinomial sampling, stopping after a specified number of lines."""
    model.eval()
    start_ids = tokenizer.encode(prompt).ids
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_ids = []
    newline_token_id = tokenizer.token_to_id("\n")
    eos_token_id = tokenizer.token_to_id('<|endoftext|>')
    lines_generated = 0
    
    for _ in range(max_new_tokens):
        block_size = model.cfg["context_length"]
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        
        logits = model(x_cond)[:, -1, :]

        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(device), logits)

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
 
        if next_id.item() == eos_token_id:
            break
        
        generated_ids.append(next_id.item())
        x = torch.cat((x, next_id), dim=1)

        print(tokenizer.decode([generated_ids[-1]]), end='')

        if next_id.item() == newline_token_id:
            lines_generated += 1
            if lines_generated >= num_lines:
                break
    
    print()
            
    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate code using a trained model.")
    parser.add_argument('--model_path', type=str, default='python_model.pth', help="Path to the trained model weights (.pth file).")
    parser.add_argument('--tokenizer_path', type=str, default='./tokenizer_training_data/custom_code_tokenizer.json', help="Path to the custom tokenizer file.")
    parser.add_argument('--num_lines', type=int, default=5, help="The number of lines to generate.")

    # Generation strategy arguments
    parser.add_argument('--max_new_tokens', type=int, default=200, help="Maximum number of tokens to generate (used with sampling).")

    args = parser.parse_args()

    # Setup device
    device = 'cpu'
    print(f"Using device: {device}", file=sys.stderr)

    # 1. Load Tokenizer
    try:
        tokenizer = CodeTokenizer.load(args.tokenizer_path)
        print(f"Tokenizer loaded successfully from {args.tokenizer_path}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Tokenizer file not found at {args.tokenizer_path}", file=sys.stderr)
        return

    # 2. Configure and Load Model
    model_config = QWEN3_CONFIG_1_7B
    model_config["vocab_size"] = tokenizer.get_vocab_size()
    model_config["context_length"] = 512
    model = Qwen3Model(model_config).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model weights loaded successfully from {args.model_path}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}. Running with an untrained model.", file=sys.stderr)
    except Exception as e:
        print(f"Error loading model weights: {e}", file=sys.stderr)
        return

    # model = torch.compile(model, fullgraph=True)

    model.eval()

    propmt = '''\
import argparse
import torch
import sys

from main import Qwen3Model, QWEN_CONFIG_06_B
from tokenizer import CodeTokenizer


@torch.no_grad()
def generate_sampling(model, tokenizer, prompt, num_lines, max_new_tokens, device):
    """Generates text using multinomial sampling, stopping after a specified number of lines."""
    model.eval()
    start_ids = tokenizer.encode(prompt).ids
    x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_ids = []
    newline_token_id = tokenizer.token_to_id("\n")
    eos_token_id = tokenizer.token_to_id('<|endoftext|>')
    lines_generated = 0
    
    for _ in range(max_new_tokens):
'''
    print("\n--- Generating with Multinomial Sampling ---", file=sys.stderr)
    generated_code = generate_sampling(
        model=model,
        tokenizer=tokenizer,
        prompt=propmt,
        num_lines=args.num_lines,
        max_new_tokens=args.max_new_tokens,
        device=device
    )
    print(generated_code)

if __name__ == '__main__':
    main()
