import argparse
import torch
import torch.optim as optim

import json
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
parser.add_argument(
    '--max-epoch', type=int, help='setting max epoch'
)

args = parser.parse_args()
device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
print(device)
input_dim = 500
classes = 8


def main():
    train_set = SinaDataset(path.join(args.source, 'train.json'), input_dim)
    test_set = SinaDataset(path.join(args.source, 'test.json'), input_dim)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=True, drop_last=True)

    # model = TextCNN(input_dim, 200)
    model = MyLSTM(input_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    
    epoch = 0
    train_loss = []
    train_accu = []
    valid_loss = []
    valid_accu = []
    while True:
        epoch += 1
        epoch_loss, epoch_accu = train_one_epoch(epoch, model, optimizer, train_loader, device, args.bs)
        val_loss, val_accu = validate(model, test_loader, device, args.bs)
        train_loss += epoch_loss
        train_accu += epoch_accu
        valid_loss += val_loss
        valid_accu += val_accu

        print('saving...')
        torch.save(model.state_dict(), './saved_models/epoch' + str(epoch) + '.pkl')
        print()

        if args.max_epoch and epoch >= args.max_epoch:
            train_result = {
                'batch-size': args.bs,
                'train-loss': train_loss,
                'train-accu': train_accu,
                'valid-loss': valid_loss,
                'valid-accu': valid_accu
            }
            with open('train-result.json', 'w', encoding='utf-8') as f:
                json.dump(train_result, f)
            
            break


if __name__ == '__main__':
    main()
