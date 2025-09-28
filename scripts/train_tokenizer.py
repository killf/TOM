from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer
import argparse
import random
import json
import os


def read_texts_from_jsonl(data_files):
    for file in data_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines()[:10000]:
                data = json.loads(line)
                yield data["text"]


def train_tokenizer(data_files: list[str]):
    print(f"Training tokenizer on data from: ")
    for file in data_files:
        print(f" - {file}")
    print("This may take a while...\n")

    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # 定义特殊token
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # 设置训练器并添加特殊token
    trainer = trainers.BpeTrainer(
        vocab_size=6400,
        special_tokens=special_tokens,  # 确保这三个token被包含
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 读取文本数据
    texts = read_texts_from_jsonl(data_files)

    # 训练tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 检查特殊token的索引
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    # 保存tokenizer
    tokenizer_dir = "model"
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

    print("Tokenizer training completed and saved.\n")


def eval_tokenizer():
    from transformers import AutoTokenizer

    # 加载预训练的tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model")

    messages = [
        {"role": "system", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"},
        {"role": "user", "content": "你来自哪里？"},
        {"role": "assistant", "content": "我来自地球"},
    ]
    new_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(new_prompt)

    # 获取实际词汇表长度（包括特殊符号）
    actual_vocab_size = len(tokenizer)
    print("tokenizer实际词表长度：", actual_vocab_size)

    model_inputs = tokenizer(new_prompt)
    print("encoder长度：", len(model_inputs["input_ids"]))

    input_ids = model_inputs["input_ids"]
    response = tokenizer.decode(input_ids, skip_special_tokens=False)
    print("decoder和原始文本是否一致：", response == new_prompt)


def main(data_files):
    train_tokenizer(data_files)
    eval_tokenizer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Tokenizer", description="Train a tokenizer")
    parser.add_argument(
        "-i", "--data-file", type=str, default=["dataset/pretrain_hq.jsonl"], nargs="+"
    )
    args = parser.parse_args()

    random.seed(2025)
    main(args.data_file)
