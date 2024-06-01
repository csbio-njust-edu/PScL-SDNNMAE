
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


class DNN_4l(nn.Module):
    def __init__(self,input_dim=256, num_classes=7):
        super(DNN_4l, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=512, bias=True)
            , nn.BatchNorm1d(512)
            , nn.ReLU(inplace=True)
            , nn.Dropout(0.25)
            , nn.Linear(in_features=512, out_features=256, bias=True)
            , nn.BatchNorm1d(256)
            , nn.ReLU(inplace=True)
            , nn.Dropout(0.25)
            , nn.Linear(in_features=256, out_features=64, bias=True)
            , nn.BatchNorm1d(64)
            , nn.ReLU(inplace=True)
            , nn.Linear(in_features=64, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

