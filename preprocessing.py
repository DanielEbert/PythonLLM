import random
import sys
import string
import random
import sys

from tokenize_rt import src_to_tokens, tokens_to_src, Token
import ast


def sanitize_input(text: str) -> str:
    """
    Replaces all non-printable ASCII characters with a single special character.
    """
    # Using string.printable which includes digits, ascii_letters, punctuation, and whitespace.
    return ''.join([char if char in string.printable else '?' for char in text])


def preprocess(source: str, dropout_chance: float = 0.5) -> str:
    source = sanitize_input(source)

    try:
        tree = ast.parse(source)
        docstring_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) and \
                    node.body and isinstance(node.body[0], ast.Expr) and \
                    isinstance(node.body[0].value, ast.Constant):
                docstring_nodes.add(node.body[0].lineno)
    except (SyntaxError, ValueError):
        # If the source is not valid Python, we can't reliably find docstrings
        docstring_nodes = set()

    try:
        tokens = src_to_tokens(source)
        new_tokens = []
        for i, token in enumerate(tokens):
            # Remove comments
            if token.name == 'COMMENT':
                continue
            if token.name == 'INDENT':
                new_tokens.append(Token(name='NAME', src='<SCOPE_IN>'))
                continue

            # Replace DEDENT with <SCOPE_OUT>
            if token.name == 'DEDENT':
                new_tokens.append(Token(name='NAME', src='<SCOPE_OUT>'))
                continue

            # A simple heuristic to remove docstrings:
            # they are string literals that are the first token on a line,
            # often following an indent or a newline.
            if (
                token.name == 'STRING' and
                token.line in docstring_nodes
            ):
                # Also remove the trailing newline to avoid excessive blank lines
                if i + 1 < len(tokens) and tokens[i + 1].name in ('NL', 'NEWLINE'):
                    if (
                        i + 2 < len(tokens) and
                        tokens[i+2].name in ('DEDENT', 'ENDMARKER')
                    ):
                        continue # keep the newline before a dedent or end of file
                    elif (
                        i > 0 and
                        tokens[i-1].name in ('INDENT')
                    ):
                        # if the docstring is the only thing in a block, we need to
                        # add a pass statement
                        new_tokens.append(Token(name='NAME', src='pass'))

                continue
        
            new_tokens.append(token)

        source = tokens_to_src(new_tokens)

        processed_lines = []
        for line in source.splitlines():
            line = line.strip()
            is_import_statement = line.startswith('import ') or line.startswith('from ')
            if is_import_statement and random.random() < dropout_chance:
                continue
            if line:
                processed_lines.append(line)
        return '\n'.join(processed_lines)
    except Exception as e:
        print('Failed to parse in preprocess:', e)
        return ''


if __name__ == '__main__':
    # Ensure a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_python_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            input_code = f.read()
 
        processed_code = preprocess(input_code, dropout_chance=0.5)
        
        print("\n--- PROCESSED CODE ---")
        print(processed_code)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
