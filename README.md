# PythonLLM

LLM trained exclusively on python code.

1. Data Preparation
    - Uses the python subset from [the-stack](https://huggingface.co/datasets/bigcode/the-stack)
    - Preprocessed to remove comments, docstrings, removal of non-printable characters (e.g. chinese characters), Indentation Transformation and 'import' dropout from [FLCC Paper](https://arxiv.org/html/2405.08704v3), remove empty lines.

2. Tokenizer
    - Based on BPE, but extended to enable character-pair encoding over whitespaces, e.g. 'for i in range(' is a token. Idea is from [FLCC Paper](http://arxiv.org/html/2405.08704v3).

3. LLM Architecture is Gwen3 from [LLMs-from-scratch repo](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3)

4. Modern training loop with LR schedulers (warmup followed by cosing annealing) and dataloaders in separate processes. Loss is CrossEntropy.

6. Inference with optional beam search.

### Install

Install Dependencies:
~~~
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
~~~

### Train

Train Tokenizer:
~~~
. venv/bin/activate
python tokenizer.py
~~~

Train Model:
~~~
. venv/bin/activate
python main.py
~~~
