#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, args):

        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(args.num_features, args.until)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(args.until, args.num_classes)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def get_nets(args):
    model = MLP(args)
    return model

