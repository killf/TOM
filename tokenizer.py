from typing import List
import json


class Tokenizer:
    def __init__(self, model_file: str):
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        vocab = json.load(open(model_file, encoding="utf8"))
        self.vocab = {char: idx + 3 for idx, char in enumerate(vocab)}
        self.vocab_idx = {idx: char for char, idx in self.vocab.items()}
        self.size = len(self.vocab) + 3
        
    def encode(self, text: str, bos: bool=False, eos: bool=False) -> List[int]:
        tokens = [self.vocab[char] for char in text]
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: List[int], bos: bool=False, eos: bool=False) -> str:
        if bos:
            tokens = tokens[1:]
        if eos:
            tokens = tokens[:-1]
        return "".join([self.vocab_idx[idx] for idx in tokens if idx in self.vocab_idx])
    

def train(data_file_list: List[str], model_file: str=None):
    for data_file in isinstance(data_file_list, list) and data_file_list or [data_file_list]:
        data = json.load(open(data_file, encoding="utf8"))
        vocab = set()
        for line in data:
            for char in line["text"]:
                vocab.add(char)

    vocab = list(vocab)
    vocab.sort()
    if model_file:
        json.dump(vocab, open(model_file, "w", encoding="utf8"), ensure_ascii=False, indent=2)
    return vocab
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog='Tokenizer', description='Train a tokenizer')
    parser.add_argument("-i", "--data-file", type=str, default="data/data.json", nargs="+")
    parser.add_argument("-o", "--vocab-file", type=str, default="data/vocab.json")
    args = parser.parse_args()
    
    train(args.data_file, args.vocab_file)