import torch
import torch.nn as nn
import nn.functional as F

embedding_dim = 300


class TextCNN(nn.Module):
    
    def __init__(self, input_dim=500, filters=20):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, (3, embedding_dim))

    def forward(x):
        pass


class MyLSTM(nn.Module):
    
    def __init__(self, input_dim=500, hidden_size=32):
        super(MyLSTM, self).__init__()

    def forward(x):
        pass