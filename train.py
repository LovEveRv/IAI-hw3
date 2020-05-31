import torch
import torch.nn.functional as F

import numpy as np
from time import time

log_interval = 1


def train_one_epoch(epoch, model, optimizer, loader, device, batch_size=128):
    print('=======TRAINING EPOCH {}======='.format(epoch))
    # We assume model has been set on device
    data_len = len(loader)
    epoch_loss = []
    epoch_accu = []
    model.train()

    start_time = time()
    for batch_id, (names, labels, texts) in enumerate(loader):
        correct = 0
        optimizer.zero_grad()
        output = model(texts)
        loss = F.nll_loss(output, labels)
        epoch_loss.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        epoch_accu.append(correct * 1.0 / batch_size)
        loss.backward()
        optimizer.step()

        if (batch_id + 1) % log_interval == 0:
            print('Batch [{}/{}]\tLoss [{:.4f}]\tAccu [{:.4f}]\tCost {:.3f} seconds'.format(
                batch_id + 1, data_len, loss.item(), correct * 1.0 / batch_size, time() - start_time
            ))
            start_time = time()

    return epoch_loss, epoch_accu


def validate(model, loader, device, batch_size=128):
    print('=======VALIDATING=======')
    data_len = len(loader)
    val_loss = []
    val_accu = []
    model.eval()

    start_time = time()
    for batch_id, (names, stories, refers) in enumerate(loader):
        correct = 0
        with torch.no_grad():
            output = model(texts)
            loss = F.nll_loss(output, labels)
        val_loss.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        val_accu.append(correct * 1.0 / batch_size)
            
    print('Average Loss [{:.4f}]\tAccuracy [{:.4f}]\tCost {:.3f} seconds'.format(
        np.mean(val_loss), np.mean(val_accu), time() - start_time
    ))
    return val_loss, val_accu
