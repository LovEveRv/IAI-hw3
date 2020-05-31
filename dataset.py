import torch
from torch.utils.data import Dataset

import json


class SinaDataset(Dataset):
    
    def __init__(self, src_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cur = self.data[index]
        return cur['id'], cur['label'], torch.tensor(cur['text']).float()