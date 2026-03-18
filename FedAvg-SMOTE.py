import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset
from utils import sampling
from utils.options import args_parser
from models.ResampleUpdate import LocalUpdate
from models import Nets
from models.Fed import FedAvg
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

if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.epochs = 50
    args.local_ep = 5
    args.model = 'MLP'
    args.dataset = 'A'
    args.seed = 11
    torch.manual_seed(args.seed)

    if args.dataset == 'Taiwan':  # 10
        args.num_users = 10
        args.num_features = 91
        args.until = 46
    elif args.dataset == 'GMSC':  # 20
        args.num_users = 20
        args.num_features = 10
        args.until = 6
        args.local_bs = 1280
    elif args.dataset == 'HMEQ':  # 10
        args.num_users = 10
        args.num_features = 55
        args.until = 28
    elif args.dataset == 'Loan Data':  # 5
        args.num_users = 5
        args.num_features = 28
        args.until = 14
        args.local_bs = 128
    elif args.dataset == 'A':  # 5
        args.num_users = 5
        args.num_features = 42
        args.until = 28
        args.local_bs = 128
    elif args.dataset == 'LC':  # 15
        args.num_users = 15
        args.num_features = 34
        args.until = 18
        args.local_bs = 512

    # load dataset, split and sample users
    dataset_train, dataset_test, dict_users = sampling.get_dict_users(args.dataset, args.num_users, args.seed)
    train_fea = torch.tensor(dataset_train.drop('target', axis=1).astype(float).values, dtype=torch.float)
    train_labels = dataset_train.loc[:, 'target'].values
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    dataset_train = TensorDataset(train_fea, train_labels)

    # build model
    Net_Glob = Nets.get_nets(args).to(args.device)
    print(Net_Glob)
    Net_Glob.train()
    # copy weights
    w_glob = Net_Glob.state_dict()

    # training
    # FedAvg_算法
    print('----------------------------运行FedAvg算法----------------------------')
    net_glob = copy.deepcopy(Net_Glob)
    fedavg_test_ba = []
    fedavg_test_f1_score = []
    fedavg_test_gmean = []
    fedavg_test_recall = []
    fedavg_train_loss = []
    avg_epoch_time = []
    for iter in range(args.epochs):
        profit_globa = []
        cost_globa = []
        loss_locals = []
        ba_glob_val = []
        gmean_glob_val = []
        f1_score_glob_val = []
        recall_glob_val = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        np.random.seed(iter)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx],Resample='SMOTE')
            w, loss = local.fedavg_train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w.state_dict()))
            # 保存这一轮参与聚合的局部模型的训练损失
            loss_locals.append(copy.deepcopy(loss))

        glob_loss_avg = sum(loss_locals) / len(loss_locals)
        # update global weights
        w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # 保存每一轮所有客户端的平均运算时间
        # 保存这一轮全局模型的测试性能
        for idx in idxs_users:
            # for idx in np.arange(args.num_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            glob_test_gmean, glob_test_ba, glob_test_f1_score, glob_test_recall, profit, cost = local.test(
                net=copy.deepcopy(net_glob))
            profit_globa.append(profit)
            cost_globa.append(cost)
            gmean_glob_val.append(glob_test_gmean)
            ba_glob_val.append(glob_test_ba)
            f1_score_glob_val.append(glob_test_f1_score)
            recall_glob_val.append(glob_test_recall)
        # print performance
        glob_AUC_ROC_avg = sum(gmean_glob_val) / len(gmean_glob_val)
        glob_AUC_PR_avg = sum(ba_glob_val) / len(ba_glob_val)
        glob_BS_plus_avg = sum(f1_score_glob_val) / len(f1_score_glob_val)
        glob_KS_avg = sum(recall_glob_val) / len(recall_glob_val)
        glob_profit = sum(profit_globa) / len(profit_globa)
        glob_cost = sum(cost_globa) / len(cost_globa)

        fedavg_train_loss.append(glob_loss_avg)
        fedavg_test_ba.append(glob_AUC_ROC_avg)
        fedavg_test_f1_score.append(glob_AUC_PR_avg)
        fedavg_test_gmean.append(glob_KS_avg)
        fedavg_test_recall.append(glob_BS_plus_avg)
    print('The final test results correspond to those reported in Table 7.')
    print('test AUC-ROC: {:.4f}\ntest AUC-PR: {:.4f}\ntest KS: {:.4f}\ntest BS+: {:.4f} '
          .format(glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg))
    # fedavg_train_loss = [round(i, 4) for i in fedavg_train_loss]
    # fedavg_test_AUC_ROC = [round(i, 4) for i in fedavg_test_ba]
    # fedavg_test_AUC_PR = [round(i, 4) for i in fedavg_test_gmean]
    # fedavg_test_KS = [round(i, 4) for i in fedavg_test_f1_score]
    # fedavg_test_BS_plus = [round(i, 4) for i in fedavg_test_recall]
    # print('FedAvg-SMOTE算法 Average Global AUC-ROC: {:.4f}'.format(sum(fedavg_test_AUC_ROC) / args.epochs))
    # print('FedAvg-SMOTE算法 Average Global AUC-PR: {:.4f}'.format(sum(fedavg_test_AUC_PR) / args.epochs))
    # print('FedAvg-SMOTE算法 Average Global KS: {:.4f}'.format(sum(fedavg_test_KS) / args.epochs))
    # print('FedAvg-SMOTE算法 Average Global BS+: {:.4f}'.format(sum(fedavg_test_BS_plus) / args.epochs))
    # fedavg_train_loss.insert(0, 'Train loss')
    # fedavg_test_AUC_ROC.insert(0, 'AUC-ROC')
    # fedavg_test_AUC_PR.insert(0, 'AUC-PR')
    # fedavg_test_KS.insert(0, 'KS')
    # fedavg_test_BS_plus.insert(0, 'BS+')
    # result = [fedavg_train_loss, fedavg_test_AUC_ROC, fedavg_test_AUC_PR, fedavg_test_KS, fedavg_test_BS_plus]
    # list2txt(result, "./save/{} FedAvg-SMOTE results.txt".format(args.dataset))

