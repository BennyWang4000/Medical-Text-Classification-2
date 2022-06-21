
from turtle import forward
import torch.nn as nn


class SVM_model(nn.Module):

    def __init__(self):
        super(SVM_model, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 2)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class text_cnn(nn.Module):
    pass
