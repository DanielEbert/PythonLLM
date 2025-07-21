import os
import string
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from prepare_tokenizer_data import create_tokenizer_training_data


class CodeTokenizer:
    """
    A prototype tokenizer for Python code that replicates the key ideas from the
    JetBrains FLCC blog post.

    Key Features:
    - Character-Pair Encoding (BPE) to learn common code fragments.
    - Custom Pre-tokenization: Allows merging over spaces/tabs but NOT newlines.
    - Constrained Training: The tokenizer is constrained to an ASCII alphabet
      from the start, ensuring the vocabulary focuses purely on code structure.
    """
    def __init__(self, vocab_size=16384):
        """
        Initializes the tokenizer components.

        Args:
            vocab_size (int): The target size for the tokenizer's vocabulary.
                              The blog mentions 16,384 for their production model.
        """
        self.vocab_size = vocab_size
        # These special tokens are mentioned in the pre-processing section of the blog post.
        self.special_tokens = ["<UNK>", "<SCOPE_IN>", "<SCOPE_OUT>"]
        self.tokenizer = self._build_tokenizer()

    def _build_tokenizer(self):
        """
        Constructs the tokenizer with BPE model and custom pre-tokenizer rules.
        """
        # 1. Initialize the core BPE model
        bpe_model = models.BPE(unk_token="<UNK>")
        tokenizer = Tokenizer(bpe_model)

        # 2. Define custom pre-tokenization rules (Insight 4)
        # The goal is to split text only by newlines and our special tokens,
        # keeping everything else together so BPE can merge across spaces/tabs.
        tokenizer.pre_tokenizer = pre_tokenizers.Split(
            pattern=r"""(<SCOPE_IN>|<SCOPE_OUT>|\n)""",
            behavior="isolated" # Treat the split delimiters as separate tokens.
        )

        # 3. Define the decoder
        tokenizer.decoder = decoders.BPEDecoder()

        return tokenizer
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()
    
    @property
    def eos_token_id(self):
        return self.tokenizer.token_to_id("<|endoftext|>")

    def train(self, file_paths):
        """
        Trains the tokenizer on a list of text files using a constrained alphabet.

        Args:
            file_paths (list): A list of paths to text files for training.
        """
        print(f"Starting training with a target vocab size of {self.vocab_size}...")

        initial_alphabet = sorted(list(set(string.printable)))

        # 1. Define the trainer with the initial alphabet constraint
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            initial_alphabet=initial_alphabet # This is the key change
        )

        # 2. Train on the files
        self.tokenizer.train(files=file_paths, trainer=trainer)
        print("Training complete.")
        print(f"Final vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text):
        """
        Encodes a string into a sequence of tokens and IDs.
        """
        return self.tokenizer.encode(text)

    def decode(self, ids):
        """
        Decodes a sequence of IDs back into a string.
        """
        return self.tokenizer.decode(ids)

    def save(self, path):
        """
        Saves the tokenizer to a file.
        """
        self.tokenizer.save(path)
        print(f"Tokenizer saved to {path}")

    @staticmethod
    def load(path):
        """
        Loads a tokenizer from a file.
        """
        loaded_tokenizer = Tokenizer.from_file(path)
        # Create a new instance of our class to wrap the loaded tokenizer
        new_instance = CodeTokenizer()
        new_instance.tokenizer = loaded_tokenizer
        new_instance.vocab_size = loaded_tokenizer.get_vocab_size()
        new_instance.special_tokens = [token for token, _ in loaded_tokenizer.get_vocab().items() if token.startswith('<') and token.endswith('>')]
        print(f"Tokenizer loaded from {path}")
        return new_instance


def main():
    """
    Main function to demonstrate the custom tokenizer's workflow.
    """
    # --- 1. Setup: Prepare preprocessed training data ---
    data_dir = "tokenizer_training_data"
    training_file = os.path.join(data_dir, "python_processed_for_tokenizer.txt")
    samples_to_process = 50000
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Create the data file. In a real project, you would only run this once.
    if not os.path.exists(training_file):
        create_tokenizer_training_data(training_file, samples_to_process)
    else:
        print(f"Found existing training data: {training_file}")

    # --- 2. Train the Tokenizer ---
    # For a real model, a vocab size of ~16k is a good starting point
    code_tokenizer = CodeTokenizer(vocab_size=16384) 
    code_tokenizer.train([training_file])

    # --- 3. Save and demonstrate the tokenizer ---
    tokenizer_path = os.path.join(data_dir, "custom_code_tokenizer.json")
    code_tokenizer.save(tokenizer_path)
    
    print("\n" + "="*50)
    print("DEMONSTRATING TOKENIZER ON PREPROCESSED TEXT")
    print("="*50 + "\n")

    # The tokenizer now expects text that has been preprocessed
    test_case = """<SCOPE_IN>
    for i in range(10):
<SCOPE_IN>
        print(f"Number: {i}")
<SCOPE_OUT>
<SCOPE_OUT>"""
    encoding = code_tokenizer.encode(test_case)
    print(f"Test Case: '{test_case}'")
    print(f" -> Tokens: {encoding.tokens}")
    print(f" -> IDs: {encoding.ids}\n")
    
    # You can now integrate this saved tokenizer ("custom_code_tokenizer.json")
    # into your main model training pipeline as described in the previous answer.


if __name__ == "__main__":
    main()
