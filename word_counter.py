"""
Count number of words in
$C2S_HOME/joss_submission/paper.md
run dir = $C2S_HOME
created by Arnab Majumdar, 29.07.2025 
"""

import re

def count_words_in_md(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        text_only = re.sub(r'[`*_>#\-\[\]()!]', '', content)
        words = text_only.split()
        return len(words)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0

# Example usage
if __name__ == "__main__":
    file_path = "./joss_submission/paper.md"  # replace with your file path
    word_count = count_words_in_md(file_path)
    print(f"Total words in '{file_path}': {word_count}")