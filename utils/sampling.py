#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
# import Data
from utils import Data

def credit_noniid(dataset, num_users,seed):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(seed)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def get_dict_users(data_name,num_users,seed):
    dataset_train, dataset_test = Data.get_dataset(data_name)
    dict_users = credit_noniid(dataset_train, num_users,seed)
    return dataset_train,dataset_test,dict_users

