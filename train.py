from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
import os

from utils import Counter, Timer, calculate_eta
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
        self.scheduler = LinearLR(self.optimizer, 0.01, 1, 1000)

        if args.model_file and os.path.exists(args.model_file):
            self.model.load_state_dict(torch.load(args.model_file, weights_only=True))
        
        self.model, self.optimizer, self.scheduler, self.train_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, self.train_loader)
        
    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                print()
                os.makedirs(self.args.outdir, exist_ok=True)                
                model = self.accelerator.unwrap_model(self.model)
                torch.save(model.state_dict(), open(os.path.join(self.args.outdir, "model.pt"), "wb"))

    def train_epoch(self, epoch):
        self.model.train()
        
        timer, counter = Timer(), Counter()        
        for step, token in enumerate(self.train_loader):
            token = token.to(self.device)
            reader_time = timer.elapsed_time()
            
            x, y = token[:, :-1], token[:, 1:]
            p, loss = self.model(x, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.scheduler.step()

            if self.accelerator.is_main_process:
                loss = float(loss)            
                batch_time = timer.elapsed_time()
                counter.append(loss=loss, reader_time=reader_time, batch_time=batch_time)
                eta = calculate_eta(len(self.train_loader) - step, counter.batch_time)
            
                print(f"[epoch={epoch + 1}/{self.args.epochs}] "
                  f"step={step + 1}/{len(self.train_loader)} "
                  f"loss={loss:.4f}/{counter.loss:.4f} "
                  f"batch_time={counter.batch_time:.4f} "
                  f"reader_time={counter.reader_time:.4f} "
                  f"| ETA {eta}",
                  end="\r",
                  flush=True)
                timer.restart()

    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog='Trainer', description='Trainer')
    parser.add_argument("--data-file", type=str, default="data/data.jsonl")
    parser.add_argument("--vocab-file", type=str, default="data/vocab.json")
    parser.add_argument("--model-file", type=str, default="output/model.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--outdir", type=str, default="output")

    args = parser.parse_args()
    
    Trainer(args).train()
