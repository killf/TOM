import torch
import os

from tokenizer import Tokenizer
from model import Transformer

class Sampler:
    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = Tokenizer(args.vocab_file)
        self.model = Transformer(self.tokenizer.size, 1024, 8, 8).to(self.device)
        self.model.load_state_dict(torch.load(args.model_file, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def sample(self, text):
        print(text, end="", flush=True)
        tokens = self.tokenizer.encode(text, True)

        while True:
            p = self.model(torch.tensor([tokens], dtype=torch.int64, device=self.device))
            t = int(torch.argmax(p))

            print(self.tokenizer.decode([t]), end="", flush=True)
            tokens.append(t)            

            if t in [self.tokenizer.eos_id, self.tokenizer.pad_id]:
                break
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog='Trainer', description='Trainer')
    parser.add_argument("--vocab-file", type=str, default="data/vocab.json")
    parser.add_argument("--model-file", type=str, default="output/model.pt")
    args = parser.parse_args()
    
    Sampler(args).sample("从前")