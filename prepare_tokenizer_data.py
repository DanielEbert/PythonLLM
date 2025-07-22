import os
from datasets import load_dataset
from tqdm import tqdm
from preprocessing import preprocess

def create_tokenizer_training_data(output_path: str, num_samples: int):
    dataset = load_dataset('bigcode/the-stack', data_dir='data/python', split='train', streaming=True).take(num_samples)

    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in tqdm(dataset, total=num_samples, desc='Processing files'):
            content = sample.get('content')
            if not content:
                content
            
            f.write(preprocess(content) + '\n')
            count += 1


if __name__ == '__main__':
    DATA_DIR = "tokenizer_training_data"
    OUTPUT_FILE = os.path.join(DATA_DIR, "python_processed_for_tokenizer.txt")
    SAMPLES_TO_PROCESS = 50000 # Increase for a better tokenizer

    os.makedirs(DATA_DIR, exist_ok=True)
    create_tokenizer_training_data(OUTPUT_FILE, SAMPLES_TO_PROCESS)
