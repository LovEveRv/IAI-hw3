import argparse
import torch
import torch.nn.functional as F

import json
import numpy as np
from time import time
from os import path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from scipy.stats import pearsonr

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
    print('=======TESTING=======')
    data_len = len(loader)
    val_loss = []
    val_accu = []
    predictions = []
    answers = []
    true_dists = []
    pred_dists = []
    model.eval()

    start_time = time()
    for batch_id, (dists, labels, texts) in enumerate(loader):
        texts = texts.to(device)
        labels = labels.to(device)

        correct = 0
        with torch.no_grad():
            output = model(texts)
            loss = F.nll_loss(output, labels)
        val_loss.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        
        predictions += pred.cpu().numpy().tolist()
        answers += labels.cpu().numpy().tolist()
        true_dists += dists.cpu().numpy().tolist()
        pred_dists += output.cpu().numpy().tolist()
        
        correct += pred.eq(labels.view_as(pred)).sum().item()
        val_accu.append(correct * 1.0 / batch_size)
            
    print('Average Loss [{:.4f}]\tAccuracy [{:.4f}]\tCost {:.3f} seconds'.format(
        np.mean(val_loss), np.mean(val_accu), time() - start_time
    ))
    return predictions, answers, pred_dists, true_dists


def calc_f1_score(pred, ans):
    format_str = """\n
    =======F1 SCORE=======
    macro:    {:.4f}
    micro:    {:.4f}
    weighted: {:.4f}
    ======================
    """.format(
        f1_score(ans, pred, average='macro'),
        f1_score(ans, pred, average='micro'),
        f1_score(ans, pred, average='weighted')
    )
    print(format_str)


def calc_coef(pred, ans):
    data_len = len(pred)
    coef = 0.
    for p, t in zip(pred, ans):
        coef += pearsonr(p, t) / data_len
    print('=======PEARSONR=======\n{:.4f}'.format(coef))


def main():
    test_set = SinaDataset(path.join(args.source, 'test.json'), input_dim)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False, drop_last=True)

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
    
    pred, ans, pred_dists, true_dists = test(model, test_loader, device, args.bs)
    calc_f1_score(pred, ans)
    calc_coef(pred_dists, true_dists)


if __name__ == '__main__':
    main()
