#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import copy
import torch

def encrypt_vector(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters
# list解密
def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters

def FedSec_Cloud(w,local_w,public_key):
    for i in range(len(w)):
        for k in w[i].keys():
            w[i][k] = w[i][k]*local_w[i]
    w_locals = w
    # 加密局部模型参数进行安全聚合
    w_shape = {}
    sum_parameters = {}
    for i in range(len(w_locals)):
        if i == 0:
            for key, var in w_locals[i].items():
                w_locals[i][key] = var.clone().detach()
                w_shape[key] = var.shape
                w_locals[i][key] = encrypt_vector(public_key, w_locals[i][key])
                sum_parameters[key] = w_locals[i][key]
        else:
            for key in w_locals[i]:
                sum_parameters[key] = np.add(sum_parameters[key], encrypt_vector(public_key, w_locals[i][key]))
    encrypt_w_glob = sum_parameters
    return encrypt_w_glob, w_shape

