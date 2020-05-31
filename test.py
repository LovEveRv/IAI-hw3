import argparse
import torch
import torch.nn.functional as F

import json
import numpy as np
from time import time
from os import path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_scores

from dataset import SinaDataset
from models import TextCNN, MyLSTM


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


def test(model, loader, device, batch_size):
    pprint('=======TESTING=======')
    data_len = len(loader)
    val_loss = []
    val_accu = []
    predictions = []
    answers = []
    model.eval()

    start_time = time()
    for batch_id, (names, labels, texts) in enumerate(loader):
        texts = texts.to(device)
        labels = labels.to(device)

        correct = 0
        with torch.no_grad():
            output = model(texts)
            loss = F.nll_loss(output, labels)
        val_loss.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        predictions += pred
        answers += labels
        correct += pred.eq(labels.view_as(pred)).sum().item()
        val_accu.append(correct * 1.0 / batch_size)
            
    print('Average Loss [{:.4f}]\tAccuracy [{:.4f}]\tCost {:.3f} seconds'.format(
        np.mean(val_loss), np.mean(val_accu), time() - start_time
    ))
    return predictions, answers


def calc_f1_score(pred, ans):
    format_str = """\n
    =======F1 SCORE=======
    macro:    {:.4f}
    micro:    {:.4f}
    weighted: {:.4f}
    ======================
    """.format(
        f1_scores(ans, pred, average='macro'),
        f1_scores(ans, pred, average='micro'),
        f1_scores(ans, pred, average='weighted')
    )
    print(format_str)


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
    
    pred, ans = test(model, test_loader, device, args.bs)
    calc_f1_score(pred, ans)


if __name__ == '__main__':
    main()
