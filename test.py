import argparse
import torch

import json
from os import path
from torch.utils.data import DataLoader

from dataset import SinaDataset
from models import TextCNN, MyLSTM
from train import validate 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', type=str, help='source directory of training and test data'
)
parser.add_argument(
    '--bs', type=int, default=128, help='batch size'
)
parser.add_argument(
    '--cuda', action='store_true', default=False, help='use cuda for training'
)
parser.add_argument(
    '--model', type=str, help='textcnn/lstm'
)

args = parser.parse_args()
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
print(device)
input_dim = 500
classes = 8


def main():
    test_set = SinaDataset(path.join(args.source, 'test.json'), input_dim)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=True, drop_last=True)

    if args.model == 'textcnn':
        model = TextCNN(input_dim, 200)
        model.load_state_dict(torch.load('./saved_models/textcnn.pkl'))
    elif args.model == 'lstm':
        model = MyLSTM(input_dim, hidden_dim=8)
        model.load_state_dict(torch.load('./saved_models/lstm.pkl'))
    else:
        print('"--model" argument only accepts "textcnn" or "lstm"')
        exit(0)
    
    model = model.to(device)
    
    validate(model, test_loader, device, args.bs)


if __name__ == '__main__':
    main()
