import torch
import torch.nn as nn
import torch.nn.functional as F

embedding_dim = 300


class TextCNN(nn.Module):
    
    def __init__(self, input_dim, filters=50, classes=8):
        super(TextCNN, self).__init__()
        self.input_dim = input_dim
        # self.filters = filters

        self.conv2 = nn.Conv2d(1, filters, (2, embedding_dim))
        self.conv3 = nn.Conv2d(1, filters, (3, embedding_dim))
        self.conv4 = nn.Conv2d(1, filters, (4, embedding_dim))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(3 * filters, classes)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim, embedding_dim)

        x2 = F.relu(self.conv2(x)).squeeze(3)
        x3 = F.relu(self.conv3(x)).squeeze(3)
        x4 = F.relu(self.conv4(x)).squeeze(3)

        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        x4 = F.max_pool1d(x4, x4.shape[2]).squeeze(2)

        x = torch.cat((x2, x3, x4), dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        out = F.log_softmax(x, dim=1)
        return out


class MyLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, dropout=0.5, classes=8):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


class MLP(nn.Module):

    def __init__(self, input_dim=500, classes=8):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        return out