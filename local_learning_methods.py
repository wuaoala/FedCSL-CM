import numpy as np
import torch
from torch.utils.data import TensorDataset
from utils import sampling
from utils.options import args_parser
from models import Nets
from models.Fed import local_result
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")
def list2txt(list, path):
    file = open(path, 'w', encoding="utf-8")
    for l in list:
        l = str(l)  # 强制转换
        if l[-1] != '\n':
            l = l + '\n'
        file.write(l)
    file.close()
    print(f"{path}文件存储成功")

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
    return profit, cost
if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.dataset = 'LC'
    args.seed = 11
    torch.manual_seed(11)

    if args.dataset == 'Taiwan':  # 10
        args.num_features = 91
        args.until = 46
        n_est = 20
        args.num_users = 10
    elif args.dataset == 'GMSC':  # 20
        args.num_features = 10
        args.until = 6
        n_est = 1
        args.num_users = 20
    elif args.dataset == 'HMEQ':  # 10
        args.num_features = 55
        args.until = 28
        n_est = 5
        args.num_users = 10
    elif args.dataset == 'Loan Data':  # 5
        args.num_features = 28
        args.until = 14
        n_est = 5
        args.num_users = 5
    elif args.dataset == 'A':  # 5
        args.num_features = 42
        args.until = 28
        n_est = 5
        args.num_users = 5
    elif args.dataset == 'LC':  # 15
        args.num_features = 34
        args.until = 18
        n_est = 3
        args.num_users = 15


    # load dataset, split and sample users
    dataset_train,dataset_test,dict_users = sampling.get_dict_users(args.dataset,args.num_users,args.seed)
    # training
    print('---------------------------各个客户端单独训练模型-----------------------------')
    idxs_users = np.arange(args.num_users)
    lr_recall = []
    lr_ba = []
    lr_f1_score = []
    lr_gmean = []
    lr_profit = []
    lr_cost = []

    svm_recall = []
    svm_ba = []
    svm_f1_score = []
    svm_gmean = []
    svm_profit = []
    svm_cost = []

    xgb_recall = []
    xgb_ba = []
    xgb_f1_score = []
    xgb_gmean = []
    xgb_profit = []
    xgb_cost = []

    mlp_recall = []
    mlp_ba = []
    mlp_f1_score = []
    mlp_gmean = []
    mlp_profit = []
    mlp_cost = []

    lgbm_recall = []
    lgbm_ba = []
    lgbm_f1_score = []
    lgbm_gmean = []
    lgbm_profit = []
    lgbm_cost = []
    for idx in idxs_users:
        idxs = list(dict_users[idx])
        np.random.seed(233)
        train_idxs = np.random.choice(idxs, round(len(idxs) * 0.8), replace=False)
        test_idxs = list(set(idxs) - set(train_idxs))
        LR = LogisticRegression()
        # RF = RandomForestClassifier(n_estimators=20,max_depth=1)
        SVM = SVC(probability=True,kernel='linear')
        XGB = XGBClassifier(n_estimators=n_est, learning_rate=0.2,max_depth=1)
        LGBM = LGBMClassifier(n_estimators=n_est,max_depth=1,verbosity=-1)
        MLP = MLPClassifier(hidden_layer_sizes=(args.until,),learning_rate_init=0.01,alpha=0.25)

        LR.fit(dataset_train.iloc[train_idxs,:-1],dataset_train.iloc[train_idxs,-1])
        y_pred = LR.predict(dataset_train.iloc[test_idxs,:-1])
        y_prob = LR.predict_proba(dataset_train.iloc[test_idxs,:-1])[:, 1]
        target = dataset_train.iloc[test_idxs,-1]
        y_prob = torch.tensor(y_prob)
        target = torch.tensor(np.array(target))
        profit, cost = compute_EMP(target, y_prob)

        auc_roc = metrics.roc_auc_score(target, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(target, y_prob)
        auc_pr = metrics.auc(recall, precision)
        label_1 = np.nonzero(target == 1)

        bs_plus = metrics.brier_score_loss(target[label_1], y_prob[label_1])
        fpr, tpr, thresholds = metrics.roc_curve(target, y_prob)
        ks = max(abs(fpr - tpr))

        lr_cost.append(cost)
        lr_profit.append(profit)
        lr_gmean.append(auc_roc)
        lr_recall.append(auc_pr)
        lr_ba.append(bs_plus)
        lr_f1_score.append(ks)

        SVM.fit(dataset_train.iloc[train_idxs, :-1], dataset_train.iloc[train_idxs, -1])
        y_pred = SVM.predict(dataset_train.iloc[test_idxs, :-1])
        y_prob = SVM.predict_proba(dataset_train.iloc[test_idxs, :-1])[:, 1]
        target = dataset_train.iloc[test_idxs, -1]
        y_prob = torch.tensor(y_prob)
        target = torch.tensor(np.array(target))
        profit, cost = compute_EMP(target, y_prob)
        svm_cost.append(cost)
        svm_profit.append(profit)
        auc_roc = metrics.roc_auc_score(target, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(target, y_prob)
        auc_pr = metrics.auc(recall, precision)
        label_1 = torch.nonzero(target == 1)
        bs_plus = metrics.brier_score_loss(target[label_1], y_prob[label_1])
        fpr, tpr, thresholds = metrics.roc_curve(target, y_prob)
        ks = max(abs(fpr - tpr))
        svm_gmean.append(auc_roc)
        svm_recall.append(auc_pr)
        svm_ba.append(bs_plus)
        svm_f1_score.append(ks)

        XGB.fit(dataset_train.iloc[train_idxs, :-1], dataset_train.iloc[train_idxs, -1])
        y_pred = XGB.predict(dataset_train.iloc[test_idxs, :-1])
        y_prob = XGB.predict_proba(dataset_train.iloc[test_idxs, :-1])[:, 1]
        target = dataset_train.iloc[test_idxs, -1]
        y_prob = torch.tensor(y_prob)
        target = torch.tensor(np.array(target))
        profit, cost = compute_EMP(target, y_prob)
        xgb_cost.append(cost)
        xgb_profit.append(profit)
        auc_roc = metrics.roc_auc_score(target, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(target, y_prob)
        auc_pr = metrics.auc(recall, precision)
        label_1 = torch.nonzero(target == 1)
        bs_plus = metrics.brier_score_loss(target[label_1], y_prob[label_1])
        fpr, tpr, thresholds = metrics.roc_curve(target, y_prob)
        ks = max(abs(fpr - tpr))
        xgb_gmean.append(auc_roc)
        xgb_recall.append(auc_pr)
        xgb_ba.append(bs_plus)
        xgb_f1_score.append(ks)


        LGBM.fit(dataset_train.iloc[train_idxs, :-1], dataset_train.iloc[train_idxs, -1])
        y_pred = LGBM.predict(dataset_train.iloc[test_idxs, :-1])
        y_prob = LGBM.predict_proba(dataset_train.iloc[test_idxs, :-1])[:, 1]
        target = dataset_train.iloc[test_idxs, -1]
        y_prob = torch.tensor(y_prob)
        target = torch.tensor(np.array(target))
        profit, cost = compute_EMP(target, y_prob)
        lgbm_cost.append(cost)
        lgbm_profit.append(profit)
        auc_roc = metrics.roc_auc_score(target, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(target, y_prob)
        auc_pr = metrics.auc(recall, precision)
        label_1 = torch.nonzero(target == 1)
        bs_plus = metrics.brier_score_loss(target[label_1], y_prob[label_1])
        fpr, tpr, thresholds = metrics.roc_curve(target, y_prob)
        ks = max(abs(fpr - tpr))
        lgbm_gmean.append(auc_roc)
        lgbm_recall.append(auc_pr)
        lgbm_ba.append(bs_plus)
        lgbm_f1_score.append(ks)

        MLP.fit(dataset_train.iloc[train_idxs, :-1], dataset_train.iloc[train_idxs, -1])
        y_pred = MLP.predict(dataset_train.iloc[test_idxs, :-1])
        y_prob = MLP.predict_proba(dataset_train.iloc[test_idxs, :-1])[:, 1]
        target = dataset_train.iloc[test_idxs, -1]
        y_prob = torch.tensor(y_prob)
        target = torch.tensor(np.array(target))
        profit, cost = compute_EMP(target, y_prob)
        mlp_cost.append(cost)
        mlp_profit.append(profit)
        auc_roc = metrics.roc_auc_score(target, y_prob)
        precision, recall, _ = metrics.precision_recall_curve(target, y_prob)
        auc_pr = metrics.auc(recall, precision)
        label_1 = torch.nonzero(target == 1)
        bs_plus = metrics.brier_score_loss(target[label_1], y_prob[label_1])
        fpr, tpr, thresholds = metrics.roc_curve(target, y_prob)
        ks = max(abs(fpr - tpr))
        mlp_gmean.append(auc_roc)
        mlp_recall.append(auc_pr)
        mlp_ba.append(bs_plus)
        mlp_f1_score.append(ks)

    LR_result = [np.mean(lr_gmean), np.mean(lr_recall) , np.mean(lr_f1_score) , np.mean(lr_ba),np.mean(lr_cost),np.mean(lr_profit)]
    SVM_result = [np.mean(svm_gmean), np.mean(svm_recall) , np.mean(svm_f1_score) ,np.mean(svm_ba),np.mean(svm_cost),np.mean(svm_profit)]
    XGB_result = [np.mean(xgb_gmean), np.mean(xgb_recall) , np.mean(xgb_f1_score), np.mean(xgb_ba),np.mean(xgb_cost),np.mean(xgb_profit)]
    LGBM_result = [np.mean(lgbm_gmean), np.mean(lgbm_recall) , np.mean(lgbm_f1_score), np.mean(lgbm_ba),np.mean(lgbm_cost),np.mean(lgbm_profit)]
    MLP_result = [np.mean(mlp_gmean) , np.mean(mlp_recall) , np.mean(mlp_f1_score), np.mean(mlp_ba),np.mean(mlp_cost),np.mean(mlp_profit)]
    LR_result, SVM_result, XGB_result, LGBM_result, MLP_result = local_result(args,LR_result, SVM_result, XGB_result, LGBM_result, MLP_result)
    LR_result = [round(i, 4) for i in LR_result]
    SVM_result = [round(i, 4) for i in SVM_result]
    XGB_result = [round(i, 4) for i in XGB_result]
    LGBM_result = [round(i, 4) for i in LGBM_result]
    MLP_result = [round(i, 4) for i in MLP_result]
    print('The final test results correspond to those reported in Table 3.')
    print('local performance：','AUC-ROC','AUC-PR','KS','BS+')
    print('LR: ', LR_result[:4])
    print('SVM: ', SVM_result[:4])
    print('XGB: ', XGB_result[:4])
    print('LGBM: ', LGBM_result[:4])
    print('MLP: ', MLP_result[:4])
    if args.dataset == 'LC':
        print('The local Economic performance results correspond to those reported in Table 13.')
        print('Cost', 'Profit')
        print('LR: ', LR_result[4:])
        print('SVM: ', SVM_result[4:])
        print('XGB: ', XGB_result[4:])
        print('LGBM: ', LGBM_result[4:])
        print('MLP: ', MLP_result[4:])



