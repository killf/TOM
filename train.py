from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
import os

from dataset import TOMDataset, collate_fn
from tokenizer import Tokenizer
from model import Transformer


class Trainer:
    def __init__(self, args):
        self.args = args

        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        
        self.tokenizer = Tokenizer(args.vocab_file)
        self.train_data = TOMDataset(args.data_file, self.tokenizer)
        self.train_loader = DataLoader(self.train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        
        self.model = Transformer(self.tokenizer.size, 1024, 8, 8).to(self.device)        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), args.lr)
        
        if args.model_file:
            self.model.load_state_dict(torch.load(args.model_file, weights_only=True))

        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader)
        
    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print()
                os.makedirs(self.args.outdir, exist_ok=True)
                model = self.accelerator.unwrap_model(self.model)
                torch.save(model.state_dict(), open(os.path.join(self.args.outdir, f"model.pt"), "wb"))
        
    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        for step, token in enumerate(self.train_loader):
            token = token.to(self.device)
            
            x, y = token[:, :-1], token[:, 1:]
            p, loss = self.model(x, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.accelerator.is_main_process:
                losses.append(float(loss))
                print(f"[{epoch}] [step={step}] loss={float(loss):.4f}/{sum(losses)/len(losses):.4f}", end="\r", flush=True)

    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog='Trainer', description='Trainer')
    parser.add_argument("--data-file", type=str, default="data/data.json")
    parser.add_argument("--vocab-file", type=str, default="data/vocab.json")
    parser.add_argument("--model-file", type=str, default="output/model.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--outdir", type=str, default="output")

    args = parser.parse_args()
    
    Trainer(args).train()
    # accelerate launch train.py