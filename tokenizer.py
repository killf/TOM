from typing import List
from enum import Enum
import json


class SPECIAL_TOKEN(Enum):
    PAD = 0
    BOS = 1
    EOS = 2

    SYSTEM_START = 3
    SYSTEM_END = 4
    USER_START = 5
    USER_END = 6
    ASSISTANT_START = 7
    ASSISTANT_END = 8

    FUNCTION_START = 9
    FUNCTION_ARGS = 10
    FUNCTION_RETURN = 11
    FUNCTION_END = 12

    ACTION_START = 13
    ACTION_ARGS = 14
    ACTION_END = 15

    THINK_START = 16
    THINK_STEP = 17
    THINK_REASONING = 18
    THINK_CONCLUSION = 19
    THINK_END = 20

    IMAGE_1 = 21
    IMAGE_2 = 22
    IMAGE_3 = 23
    IMAGE_4 = 24
    IMAGE_5 = 25
    IMAGE_6 = 26
    IMAGE_7 = 27
    IMAGE_8 = 28

    IMAGE_START = 29
    IMAGE_END = 30

    DIFFUSION_START = 31
    DIFFUSION_ARGS = 32
    DIFFUSION_END = 32


class Tokenizer:
    def __init__(self, model_file: str):
        vocab = json.load(open(model_file, encoding="utf8"))
        self.vocab = {char: idx + 100 for idx, char in enumerate(vocab)}
        self.vocab_idx = {idx: char for char, idx in self.vocab.items()}
        self.size = len(self.vocab) + 100
        
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
    

def train(data_file_list: List[str], model_file: str=None, use_default: bool=False) -> List[str]:
    for data_file in isinstance(data_file_list, list) and data_file_list or [data_file_list]:
        data = json.load(open(data_file, encoding="utf8"))
        vocab = set()
        for line in data:
            for char in line["text"]:
                vocab.add(char)

    if use_default:
        chinese = json.load(open("data/default.json", encoding="utf8"))
        vocab.update(chinese)

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
    parser.add_argument("-d", "--use-default", action="store_true")
    args = parser.parse_args()
    
    train(args.data_file, args.vocab_file, args.use_default)