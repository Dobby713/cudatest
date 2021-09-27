import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # model = [nn.MaxPool2d]
        self.Conv1 = nn.Conv2d(512, 128, (3, 3), (2, 2), padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.Linear1 = nn.Linear(16 * 128, 1024)
        self.Linear2 = nn.Linear(1024, 1)
        # self.View1 = nn.View(-1)
        # self.model = nn.Sequential(*model)

    def forward(self, X):
        h = self.Conv1(X)
        # h = F.max_pool2d(X, kernel_size=2, stride=2)
        h = self.maxpool(h)
        h = h.view(-1, 16 * 128)
        #print(h.size())
        h = F.relu(self.Linear1(h))
        # h = F.dropout(h, 0.5)
        h = F.sigmoid(self.Linear2(h))
        return h
