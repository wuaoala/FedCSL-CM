#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import math
import copy
import numpy as np
import random
from sklearn import metrics
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def compute_profit(y_true, y_pred, threshold,
                   loan_amount=100000, interest_income=10000, LGD=0.45):
    """
    计算给定阈值下的平均利润
    y_true: 实际标签 (0=违约, 1=不违约)
    y_pred: 模型预测概率
    """
    y_hat = (y_pred >= threshold).type(torch.IntTensor)  # 1=不违约, 0=违约
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_hat).ravel()
    C_FP = interest_income  # 错拒好客户的机会成本
    C_FN = loan_amount * LGD  # 放贷坏客户的违约损失
    # 成本
    total_cost = fp * C_FP + fn * C_FN
    profit_TN = interest_income
    profit_FP = -interest_income
    profit_FN = -loan_amount * LGD
    profit_TP = 0

    total_profit = (tn * profit_TN +
                    fp * profit_FP +
                    fn * profit_FN +
                    tp * profit_TP)

    avg_cost = total_cost / len(y_true)  # 平均成本
    avg_profit = total_profit / len(y_true)  # 平均利润

    return avg_cost, avg_profit

def compute_EMP(y_true, y_pred,
                loan_amount=100000, interest_income=10000, LGD=0.45, num_thresholds=100):
    """
    计算 EMP (Expected Maximum Profit)
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    cost, profit = compute_profit(y_true, y_pred, 0.5, loan_amount, interest_income, LGD)
    profits = [compute_profit(y_true, y_pred, t, loan_amount, interest_income, LGD)[1]
               for t in thresholds]
    emp = np.max(profits)
    best_threshold = thresholds[np.argmax(profits)]
    return emp, profit, cost

def kdistillation(y, labels, teacher_scores,
                  temp, alpha):
    return F.mse_loss(F.softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1))*(alpha)\
           + F.cross_entropy(y, labels) * (1 - alpha)

class FocalLoss(torch.nn.Module):

    def __init__(self, gamma, alpha, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=-1)[:, 1]  # softmax获取概率 softmax激活函数 + CE损失函数

        #在原始ce上增加动态权重因子
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = idxs
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.idxs = list(idxs)
        np.random.seed(233)
        self.train_idxs = np.random.choice(self.idxs, round(len(idxs) * 0.8), replace=False)
        self.test_idxs = list(set(self.idxs) - set(self.train_idxs))
        self.args = args
        self.loss_func = F.cross_entropy
        self.selected_clients = []

        self.ldr_train = DataLoader(DatasetSplit(dataset, self.train_idxs), batch_size=self.args.local_bs,shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset, self.test_idxs), batch_size=self.args.bs, shuffle=True)

    def fedcsl_cm_train(self, local_net, kd_net,gamma, alpha, lamda):
        local_net.train()
        # train and update
        optimizer = torch.optim.SGD(local_net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        AUC_PR = []
        AUC_ROC = []
        BS_Plus = []
        KS = []
        temp = 1
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, target) in enumerate(self.ldr_train):
                images, target = images.to(self.args.device), target.to(self.args.device)
                optimizer.zero_grad()
                log_probs = local_net(images)
                focalloss = FocalLoss(gamma=gamma, alpha=alpha)
                teacher_probs = kd_net(images).detach()
                loss_teacher = F.mse_loss(F.softmax(log_probs / temp, dim=1),
                                          F.softmax(teacher_probs / temp, dim=1))
                loss_focal = focalloss(log_probs, target)
                loss = lamda * loss_teacher + (1 - lamda) * loss_focal

                # 评估性能
                y_prob = F.softmax(log_probs, dim=1)[:, 1].cpu()
                y_prob, target = y_prob.to("cpu"), target.to("cpu")
                # try:
                #     aucroc = metrics.roc_auc_score(target.cpu().numpy(), y_prob.detach().numpy())
                # except ValueError:
                #     pass # 或者采取其他措施，例如跳过该步骤
                # AUC_ROC.append(aucroc)
                AUC_ROC.append(metrics.roc_auc_score(target.cpu().numpy(), y_prob.detach().numpy()))

                precision, recall, _ = metrics.precision_recall_curve(target.detach().numpy(), y_prob.detach().numpy())
                auc_pr = metrics.auc(recall, precision)
                AUC_PR.append(auc_pr)

                label_1 = torch.nonzero(target == 1)
                bs_plus = metrics.brier_score_loss(target[label_1].detach().numpy(), y_prob[label_1].detach().numpy())

                BS_Plus.append(bs_plus)
                fpr, tpr, thresholds = metrics.roc_curve(target.detach().numpy(), y_prob.detach().numpy())
                ks = max(abs(fpr - tpr))
                KS.append(ks)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        val_AUC_ROC = np.mean(AUC_ROC)
        val_AUC_PR = np.mean(AUC_PR)
        val_BS_Plus = np.mean(BS_Plus)
        val_KS = np.mean(KS)
        return copy.deepcopy(local_net), sum(epoch_loss) / len(epoch_loss), \
            val_AUC_ROC, val_AUC_PR, 1-val_BS_Plus , val_KS

    def focal_train(self, net,gamma, alpha, lamda):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        AUC_PR = []
        AUC_ROC = []
        BS_Plus = []
        KS = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, target) in enumerate(self.ldr_train):
                images, target = images.to(self.args.device), target.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                focalloss = FocalLoss(gamma=gamma, alpha=alpha)
                loss = focalloss(log_probs, target)
                # 评估性能
                y_prob = F.softmax(log_probs, dim=1)[:, 1].cpu()
                y_prob, target = y_prob.to("cpu"), target.to("cpu")
                AUC_ROC.append(metrics.roc_auc_score(target.cpu().numpy(), y_prob.detach().numpy()))
                precision, recall, _ = metrics.precision_recall_curve(target.detach().numpy(), y_prob.detach().numpy())
                auc_pr = metrics.auc(recall, precision)
                AUC_PR.append(auc_pr)
                label_1 = torch.nonzero(target == 1)
                bs_plus = metrics.brier_score_loss(target[label_1].detach().numpy(), y_prob[label_1].detach().numpy())
                BS_Plus.append(bs_plus)
                fpr, tpr, thresholds = metrics.roc_curve(target.detach().numpy(), y_prob.detach().numpy())
                ks = max(abs(fpr - tpr))
                KS.append(ks)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        val_AUC_ROC = np.mean(AUC_ROC)
        val_AUC_PR = np.mean(AUC_PR)
        val_BS_Plus = np.mean(BS_Plus)
        val_KS = np.mean(KS)
        return copy.deepcopy(net), sum(epoch_loss) / len(epoch_loss), \
            val_AUC_ROC, val_AUC_PR, 1 - val_BS_Plus, val_KS


    def test(self, net):
        net.eval()
        # testing
        AUC_PR = []
        AUC_ROC = []
        BS_Plus = []
        KS = []
        for batch_idx, (data, target) in enumerate(self.ldr_test):
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = net(data)
            # 评估性能

            y_prob = F.softmax(log_probs, dim=1)[:, 1].cpu()
            y_prob, target = y_prob.to("cpu"), target.to("cpu")
            emp, profit, cost = compute_EMP(target, y_prob)

            AUC_ROC.append(metrics.roc_auc_score(target.cpu().numpy(), y_prob.detach().numpy()))
            precision, recall, _ = metrics.precision_recall_curve(target.detach().numpy(), y_prob.detach().numpy())
            auc_pr = metrics.auc(recall, precision)
            AUC_PR.append(auc_pr)
            label_1 = torch.nonzero(target == 1)
            bs_plus = metrics.brier_score_loss(target[label_1].detach().numpy(), y_prob[label_1].detach().numpy())
            BS_Plus.append(bs_plus)

            fpr, tpr, thresholds = metrics.roc_curve(target.detach().numpy(), y_prob.detach().numpy())
            ks = max(abs(fpr - tpr))
            KS.append(ks)

        val_AUC_ROC = np.mean(AUC_ROC)
        val_AUC_PR = np.mean(AUC_PR)
        val_BS_Plus = np.mean(BS_Plus)
        val_KS = np.mean(KS)
        return val_AUC_ROC, val_AUC_PR, val_BS_Plus,  val_KS, profit, cost




