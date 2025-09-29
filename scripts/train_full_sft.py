from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch import optim, nn
import argparse
import warnings
import torch
import time
import math
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_base import TOMConfig, TOMForCausalLM
from dataset import SFTDataset

warnings.filterwarnings("ignore")


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        res = model(X)
        loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
            Y.size()
        )

        loss = (loss * loss_mask).sum() / loss_mask.sum()
        loss += res.aux_loss
        loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            print(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

            if wandb is not None:
                wandb.log(
                    {
                        "loss": loss * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60
                        - spend_time // 60,
                    }
                )

        if (step + 1) % args.save_interval == 0:
            model.eval()
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = TOMForCausalLM(lm_config)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    print(
        f"LLM训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M"
    )
    model = model.to(args.device)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOM Full SFT")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="TOM-Full-SFT")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-hidden-layers", default=8, type=int)
    parser.add_argument("--max-seq-len", default=512, type=int)
    parser.add_argument("--use-moe", default=False, type=bool)
    parser.add_argument("--data-path", type=str, default="dataset/sft_mini_512.jsonl")

    args = parser.parse_args()

    lm_config = TOMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"TOM-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    base_seed = 2025
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
