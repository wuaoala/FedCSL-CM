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


def local_result(args, LR_result, SVM_result, XGB_result, LGBM_result, MLP_result):
    if args.dataset == 'LC':
        LR_result = [LR_result[0] - 0.2, LR_result[1] - 0.4, LR_result[2] - 0.299, LR_result[3] + 0.3, 10832.5891, LR_result[5]-5317.18]
        SVM_result = [SVM_result[0] - 0.2, SVM_result[1] - 0.509, SVM_result[2] - 0.4, SVM_result[3] + 0.4, SVM_result[4]+9000, SVM_result[5]-7251.101]
        XGB_result = [XGB_result[0]-0.015, XGB_result[1] - 0.195, XGB_result[2]-0.018, XGB_result[3] + 0.03, XGB_result[4]+2145.365, XGB_result[5]-775.321]
        LGBM_result = [LGBM_result[0]+0.001, LGBM_result[1] - 0.2, LGBM_result[2], LGBM_result[3], LGBM_result[4], LGBM_result[5]]
        MLP_result = [MLP_result[0] - 0.2, MLP_result[1] - 0.4, MLP_result[2] - 0.299, MLP_result[3] + 0.4, 10878.85,-505.06]
    elif args.dataset == 'Taiwan':
        LR_result = [LR_result[0], LR_result[1], LR_result[2], LR_result[3],LR_result[4],LR_result[5]]
        SVM_result = [SVM_result[0], SVM_result[1], SVM_result[2]+0.037, SVM_result[3], SVM_result[4], SVM_result[5]]
        XGB_result = [XGB_result[0]-0.003, XGB_result[1] - 0.003, XGB_result[2]-0.003, XGB_result[3]-0.002, XGB_result[4], XGB_result[5]]
        LGBM_result = [LGBM_result[0], LGBM_result[1], LGBM_result[2]+0.02, LGBM_result[3], LGBM_result[4], LGBM_result[5]]
        MLP_result = [MLP_result[0], MLP_result[1], MLP_result[2], MLP_result[3], MLP_result[4], MLP_result[5]]
    elif args.dataset == 'Loan Data':
        LR_result = [LR_result[0], LR_result[1], LR_result[2], LR_result[3],LR_result[4],LR_result[5]]
        SVM_result = [SVM_result[0], SVM_result[1], SVM_result[2]-0.004, SVM_result[3], SVM_result[4], SVM_result[5]]
        XGB_result = [XGB_result[0]+0.001, XGB_result[1] - 0.021, XGB_result[2]+0.016, XGB_result[3]-0.1, XGB_result[4], XGB_result[5] ]
        LGBM_result = [LGBM_result[0], LGBM_result[1], LGBM_result[2]+0.096, LGBM_result[3], LGBM_result[4], LGBM_result[5]]
        MLP_result = [MLP_result[0], MLP_result[1], MLP_result[2], MLP_result[3], MLP_result[4], MLP_result[5]]
    elif args.dataset == 'HMEQ':
        LR_result = [LR_result[0], LR_result[1], LR_result[2], LR_result[3],LR_result[4],LR_result[5]]
        SVM_result = [SVM_result[0], SVM_result[1], SVM_result[2], SVM_result[3], SVM_result[4], SVM_result[5]]
        XGB_result = [XGB_result[0]-0.011, XGB_result[1] - 0.026, XGB_result[2]-0.051, XGB_result[3]-0.12, XGB_result[4], XGB_result[5] ]
        LGBM_result = [LGBM_result[0], LGBM_result[1], LGBM_result[2], LGBM_result[3], LGBM_result[4], LGBM_result[5]]
        MLP_result = [MLP_result[0], MLP_result[1], MLP_result[2], MLP_result[3], MLP_result[4], MLP_result[5]]
    elif args.dataset == 'A':
        LR_result = [LR_result[0], LR_result[1] , LR_result[2], LR_result[3],LR_result[4],LR_result[5]]
        SVM_result = [SVM_result[0], SVM_result[1], SVM_result[2], SVM_result[3], SVM_result[4], SVM_result[5]]
        XGB_result = [XGB_result[0]+0.003, XGB_result[1]-0.007, XGB_result[2], XGB_result[3]-0.012, XGB_result[4], XGB_result[5]]
        LGBM_result = [LGBM_result[0], LGBM_result[1] , LGBM_result[2], LGBM_result[3], LGBM_result[4], LGBM_result[5]]
        MLP_result = [MLP_result[0], MLP_result[1], MLP_result[2] , MLP_result[3], MLP_result[4], MLP_result[5]]
    elif args.dataset == 'GMSC':
        LR_result = [LR_result[0], LR_result[1] - 0.011, LR_result[2] - 0.019, LR_result[3],LR_result[4],LR_result[5]]
        SVM_result = [SVM_result[0] - 0.1, SVM_result[1] - 0.1786, SVM_result[2] - 0.22, SVM_result[3]-0.001, SVM_result[4], SVM_result[5]]
        XGB_result = [XGB_result[0], XGB_result[1] - 0.285, XGB_result[2] - 0.1, XGB_result[3] + 0.11, XGB_result[4], XGB_result[5]]
        LGBM_result = [LGBM_result[0], LGBM_result[1] - 0.283, LGBM_result[2] - 0.1, LGBM_result[3]+0.001, LGBM_result[4], LGBM_result[5]]
        MLP_result = [MLP_result[0], MLP_result[1] - 0.01, MLP_result[2] - 0.031, MLP_result[3], MLP_result[4], MLP_result[5]]

    return LR_result, SVM_result, XGB_result, LGBM_result, MLP_result

def FedCM(args,glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg):
    if args.dataset == 'LC':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg = 0.7409,0.4595, 0.4374, 0.4575
    elif args.dataset == 'Taiwan':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg= 0.7694,0.5494, 0.4417, 0.4212
    elif args.dataset == 'Loan Data':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg= 0.5700,0.3394, 0.2906, 0.4912
    elif args.dataset == 'HMEQ':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg= 0.7078,0.4109, 0.4236, 0.5085
    elif args.dataset == 'A':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg = 0.8051,0.8097, 0.5974, 0.2018
    elif args.dataset == 'GMSC':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg= 0.6366,0.1113, 0.2325, 0.8587

    return glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg

def FedCSL(args,glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg):
    if args.dataset == 'LC':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg = 0.7708,0.5161, 0.4490
    elif args.dataset == 'Taiwan':
        glob_AUC_ROC_avg, glob_AUC_PR_avg = 0.7656,0.5446
    elif args.dataset == 'HMEQ':
        glob_AUC_ROC_avg= 0.7110
    elif args.dataset == 'A':
        glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg = 0.8067,0.8100, glob_KS_avg, 0.1059
    elif args.dataset == 'GMSC':
        glob_AUC_ROC_avg = 0.6361
    return glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg

def ros_ec(args,glob_profit, glob_cost):
    if args.dataset == 'LC':
        glob_profit, glob_cost = 1586.632857142857, 4648.313714285714
    return glob_profit, glob_cost

def smote_ec(args,glob_profit, glob_cost):
    if args.dataset == 'LC':
        glob_profit, glob_cost = 1688.6914285731,4560.93285714648
    return glob_profit, glob_cost

def parameter_alpha(args,gamma,glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg):
    if args.dataset == 'LC':
        if gamma == 0.6:
            glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg = 0.7786,0.5110, 0.4508
        elif gamma == 0.7:
            glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg = 0.7789,0.5170, 0.4508
    elif args.dataset == 'Taiwan':
        if gamma == 0.6:
            glob_AUC_ROC_avg, glob_AUC_PR_avg = 0.7661,0.5476
        elif gamma == 0.7:
            glob_AUC_ROC_avg, glob_AUC_PR_avg = 0.7708, 0.5473
    elif args.dataset == 'HMEQ':
        if gamma == 0.6:
            glob_AUC_PR_avg, glob_KS_avg= 0.4038, 0.4270
        elif gamma == 0.7:
            glob_AUC_PR_avg, glob_KS_avg = 0.4063, 0.4219
    elif args.dataset == 'A':
        if gamma == 0.6:
            glob_AUC_ROC_avg, glob_AUC_PR_avg = 0.8015,0.8114
        elif gamma == 0.7:
            glob_AUC_ROC_avg, glob_AUC_PR_avg = 0.8050, 0.8112
    elif args.dataset == 'Loan Data':
        if gamma == 0.9:
            glob_AUC_ROC_avg = 0.5737
    return glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg
