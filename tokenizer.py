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

    VOICE_START = 21
    VOICE_CONTENT = 22
    VOICE_END = 23

    AUDIO_START = 24
    AUDIO_END = 25

    IMAGE_START = 26
    IMAGE_END = 27

    VIDEO_START = 28
    VIDEO_END = 29

    AUDIO_PLACEHOLDER_START = 30
    AUDIO_PLACEHOLDER_END = 31

    IMAGE_PLACEHOLDER_START = 32
    IMAGE_PLACEHOLDER_END = 33

    VIDEO_PLACEHOLDER_START = 34
    VIDEO_PLACEHOLDER_END = 35

    AUDIO_OUTPUT_START = 36
    AUDIO_OUTPUT_END = 37

    IMAGE_OUTPUT_START = 38
    IMAGE_OUTPUT_END = 39

    VIDEO_OUTPUT_START = 40
    VIDEO_OUTPUT_END = 41

    AUDIO_1 = 42
    AUDIO_2 = 43
    AUDIO_3 = 44
    AUDIO_4 = 45
    AUDIO_5 = 46
    AUDIO_6 = 47
    AUDIO_7 = 48
    AUDIO_8 = 49

    IMAGE_1 = 50
    IMAGE_2 = 51
    IMAGE_3 = 52
    IMAGE_4 = 53
    IMAGE_5 = 54
    IMAGE_6 = 55
    IMAGE_7 = 56
    IMAGE_8 = 57

    VIDEO_1 = 58
    VIDEO_2 = 59
    VIDEO_3 = 60
    VIDEO_4 = 61
    VIDEO_5 = 62
    VIDEO_6 = 63
    VIDEO_7 = 64
    VIDEO_8 = 65

    FUNCTION_DEF_START = 66
    FUNCTION_DEF_END = 67

    ACTION_DEF_START = 68
    ACTION_DEF_END = 69


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