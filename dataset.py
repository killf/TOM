from torch.utils.data import Dataset, DataLoader
import torch
import json

class TOMDataset(Dataset):
    def __init__(self, data_file: str, tokenizer = None):
        self.data = json.load(open(data_file, encoding="utf8"))
        if tokenizer is not None:
            for item in self.data:
                item["token"] = tokenizer.encode(item["text"], True, True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item

def collate_fn(data):
    result, max_len = [], 0
    for item in data:
        token = item["token"]
        result.append(token)
        max_len = max(max_len, len(token))
    
    for item in result:
        if max_len > len(item):
            item.extend([0] * (max_len - len(item)))
    
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