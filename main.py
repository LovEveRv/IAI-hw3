import argparse
import torch
import torch.optim as optim

from os import path
from torch.utils.data import DataLoader

from dataset import SinaDataset
from models import TextCNN, MyLSTM
from train import train_one_epoch, validate 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source', type=str, help='source directory of training and test data'
)
parser.add_argument(
    '--bs', type=int, default=128, help='batch size'
)
parser.add_argument(
    '--lr', type=float, default=2e-4, help='learning rate'
)
parser.add_argument(
    '--wd', type=float, default=1e-5, help='weight decay'
)
parser.add_argument(
    '--cuda', action='store_true', default=False, help='use cuda for training'
)

args = parser.parse_args()
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
print(device)
input_dim = 500
classes = 8


def main():
    train_set = SinaDataset(path.join(args.source, 'train.json'), input_dim)
    test_set = SinaDataset(path.join(args.source, 'test.json'), input_dim)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=True)

    model = TextCNN()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, args.wd)
    
    epoch = 0
    while True:
        epoch += 1
        train_one_epoch(epoch, model, optimizer, train_loader, device, args.bs)
        validate(model, test_loader, device, args.bs)

        print('saving...')
        torch.save(model.state_dict(), './saved_models/epoch' + str(epoch) + '.pkl')
        print()


if __name__ == '__main__':
    main()
