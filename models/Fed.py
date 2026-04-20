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

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def FedSec_Avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedCloud(w,local_w):
    w_cloud = copy.deepcopy(w[0])
    for k in w_cloud.keys():
        w_cloud[k] = w_cloud[k] * local_w[0]
        for i in range(1, len(w)):
            w_cloud[k] += w[i][k]*local_w[i]
    return w_cloud


def FedSec_Cloud(w,local_w,public_key):
    for i in range(len(w)):
        for k in w[i].keys():
            w[i][k] = w[i][k] * local_w[i]
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

def FedNova(w):
    w_avg = copy.deepcopy(w[0])
    w_avg_para_norm = torch.norm(torch.cat([x.view(-1) for x in w_avg.values()]))
    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], w_avg_para_norm)
        for i in range(1, len(w)):
            para_norm = torch.norm(torch.cat([x.view(-1) for x in w[i].values()]))
            w_avg[k] += torch.div(w[i][k], para_norm)
    return w_avg
def SCAFFOLD(C):
    c_avg = C[0]
    for i in range(1, len(C)):
        c_avg += C[i]
    c_avg = [torch.div(i,len(C)) for i in c_avg]
    return c_avg

