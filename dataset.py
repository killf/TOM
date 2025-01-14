from torch.utils.data import Dataset, DataLoader
import torch
import json

class TOMDataset(Dataset):
    def __init__(self, data_file: str, tokenizer = None):
        lines = open(data_file, encoding="utf8").read().split("\n")
        self.data = [json.loads(txt) for txt in lines]
        if tokenizer is not None:
            for item in self.data:
                item["token"] = tokenizer.encode(item["text"], True, True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]


def filter_by_max_tokens(data, max_tokens):
    while len(data) > 1:
        max_len = 0
        for item in data:
            max_len = max(max_len, len(item["token"]))

        if max_len * len(data) > max_tokens:
            data.pop(-1)
        else:
            return data
    return data


def collate_fn(data):
    data = filter_by_max_tokens([item for item in data], 1024 * 48)

    result, max_len = [], 0
    for item in data:
        token = item["token"]
        result.append(token)
        max_len = max(max_len, len(token))
    
    for token in result:
        if max_len > len(token):
            token.extend([0] * (max_len - len(token)))
    
    return torch.tensor(result, dtype=torch.int64)

if __name__ == "__main__":
    from tokenizer import Tokenizer

    tokenizer = Tokenizer("data/vocab.json")
    dataset = TOMDataset("data/data.json", tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    for item in loader:
        print(item)
        print(item.shape)
        break