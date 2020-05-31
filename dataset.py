import torch
from torch.utils.data import Dataset

import json


def pad_to_len(tensor, length):
    if length == -1:
        return tensor  # no processing
    
    cur_len = tensor.shape[0]
    if cur_len > length:
        return tensor[:length, :]  # cut down
    
    embed_dim = tensor.shape[1]
    zeros = torch.zeros((length - cur_len, embed_dim))  # pad
    return torch.cat((tensor, zeros), dim=0)


class SinaDataset(Dataset):
    
    def __init__(self, src_path, length=-1):
        self.length = length
        with open(src_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cur = self.data[index]
        dist = torch.tensor(cur['dist']).float()
        text = torch.tensor(cur['text']).float()
        text = pad_to_len(text, self.length)
        return dist, cur['label'], text