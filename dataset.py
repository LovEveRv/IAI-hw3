import torch
from torch.utils.data import Dataset

import json


def pad_to_len(tensor, length):
    cur_len = tensor.shape[0]
    embed_dim = tensor.shape[1]
    zeros = torch.zeros((length - cur_len, embed_dim))
    return torch.cat((tensor, zeros), dim=0)


class SinaDataset(Dataset):
    
    def __init__(self, src_path, length):
        self.length = length
        with open(src_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cur = self.data[index]
        text = torch.tensor(cur['text']).float()
        text = pad_to_len(text, self.length)
        return cur['id'], cur['label'], text