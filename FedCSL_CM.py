import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset
from utils import sampling
from utils.options import args_parser
from models.FocalUpdate import LocalUpdate
from models import Nets
from models.Fed import FedAvg,FedCloud,FedSec_Cloud
from models.Cloud_model import Cloud_evaluater
import time
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
    args.dataset = 'Taiwan'
    args.seed = 11
    torch.manual_seed(11)

    if args.dataset == 'Taiwan':  # 10
        args.num_users = 10
        args.num_features = 91
        args.until = 46
    # elif args.dataset == 'GMSC':  # 20
    #     args.num_users = 20
    #     args.num_features = 67
    #     args.until = 34
    #     args.local_bs = 512
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


    # load dataset, split and sample users
    dataset_train,dataset_test,dict_users = sampling.get_dict_users(args.dataset,args.num_users,args.seed)
    train_fea = torch.tensor(dataset_train.drop('target',axis=1).astype(float).values, dtype=torch.float)
    train_labels = dataset_train.loc[:,'target'].values
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    dataset_train = TensorDataset(train_fea,train_labels)

    # build model
    Net_Glob = Nets.get_nets(args).to(args.device)
    print(Net_Glob)
    Net_Glob.train()
    # copy weights
    w_glob = Net_Glob.state_dict()

    # training
    print('----------------------------运行FedCSL-CM算法----------------------------')
    net_local = copy.deepcopy(Net_Glob)
    net_glob = copy.deepcopy(Net_Glob)

    fedavg_test_ba = []
    fedavg_test_f1_score = []
    fedavg_test_gmean = []
    fedavg_test_recall = []
    fedavg_train_loss = []
    Cloud_evaluater = Cloud_evaluater()
    avg_epoch_time = []
    for iter in range(args.epochs):
        loss_locals = []

        ba_glob_val = []
        gmean_glob_val = []
        f1_score_glob_val = []
        recall_glob_val = []
        All_performance = []
        avg_client_time = []
        N = []
        w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        np.random.seed(iter)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        start_time = time.time()
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if iter == 0:
                net, loss,val_gmean, val_ba, val_f1score, val_recall = local.focal_train(net=copy.deepcopy(net_local).to(args.device))
            else:
                net_local.load_state_dict(torch.load('./model_param/local_net_{}.pkl'.format(idx)))
                net, loss,val_gmean, val_ba, val_f1score, val_recall = local.fedcsl_cm_train(local_net=copy.deepcopy(net_local).to(args.device),
                                           kd_net=copy.deepcopy(kd_net).to(args.device))
            # 保存局部模型参数
            w_locals.append(copy.deepcopy(net.state_dict()))
            torch.save(net.state_dict(), './model_param/local_net_{}.pkl'.format(idx))
            # 保存这一轮参与聚合的局部模型的训练损失
            loss_locals.append(copy.deepcopy(loss))
            # 计算本轮局部模型的性能
            performance_local = [val_gmean, val_ba, val_f1score, val_recall]
            # performance_local = [val_ba, val_f1score]
            All_performance.append(performance_local)
            N_local = Cloud_evaluater.Cloud_compute(performance_local)
            N.append(N_local)

        glob_loss_avg = sum(loss_locals) / len(loss_locals)
        PN, NN = Cloud_evaluater.Max_value_compute(All_performance)
        PN = Cloud_evaluater.Cloud_compute(PN)
        NN = Cloud_evaluater.Cloud_compute(NN)
        # 计算两个云的模糊贴近度
        Ps = []
        Ns = []
        for i in range(len(N)):
            Ps_i = Cloud_evaluater.Fuzzy_nearness_compute(N[i], PN)
            Ps_i = round(Ps_i, 4)
            Ps.append(Ps_i)
            Ns_i = Cloud_evaluater.Fuzzy_nearness_compute(N[i], NN)
            Ns_i = round(Ns_i, 4)
            Ns.append(Ns_i)
        # 保存局部模型的聚合权重
        local_weight = Cloud_evaluater.Assign_weight(Ps, Ns)
        # update global weights
        w_glob = FedCloud(w_locals, local_weight)
        # w_glob = FedAvg(w_locals)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        kd_net = copy.deepcopy(net_glob)
        # 保存每一轮所有客户端的平均运算时间
        end_time = time.time()
        epoch_time = end_time - start_time
        avg_epoch_time.append(epoch_time)
        # 保存这一轮全局模型的测试准确率
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            glob_test_gmean, glob_test_ba, glob_test_f1_score, glob_test_recall = local.test(net=copy.deepcopy(net_glob))
            gmean_glob_val.append(glob_test_gmean)
            ba_glob_val.append(glob_test_ba)
            f1_score_glob_val.append(glob_test_f1_score)
            recall_glob_val.append(glob_test_recall)
        # print performance
        glob_gmean_avg = sum(gmean_glob_val) / len(gmean_glob_val)
        glob_ba_avg = sum(ba_glob_val) / len(ba_glob_val)
        glob_f1_score_avg = sum(f1_score_glob_val) / len(f1_score_glob_val)
        glob_recall_avg = sum(recall_glob_val) / len(recall_glob_val)
        print('\nGlobal Round {:3d}, test AUC-ROC: {:.4f} test AUC-PR: {:.4f} test BS+: {:.4f} test KS: {:.4f}'
              .format(iter, glob_gmean_avg, glob_ba_avg, glob_f1_score_avg, glob_recall_avg))
        fedavg_train_loss.append(glob_loss_avg)
        fedavg_test_ba.append(glob_ba_avg)
        fedavg_test_f1_score.append(glob_f1_score_avg)
        fedavg_test_gmean.append(glob_gmean_avg)
        fedavg_test_recall.append(glob_recall_avg)

    # fedavg_local_test_ba = [round(i, 4) for i in fedavg_local_test_ba]
    # fedavg_local_test_gmean = [round(i, 4) for i in fedavg_local_test_gmean]
    # fedavg_local_test_f1_score = [round(i, 4) for i in fedavg_local_test_f1_score]
    # fedavg_local_test_recall = [round(i, 4) for i in fedavg_local_test_recall]
    # print('FedCSL算法 Average Local G-mean: {:.4f}'.format(sum(fedavg_local_test_gmean) / args.epochs))
    # print('FedCSL算法 Average Local BA: {:.4f}'.format(sum(fedavg_local_test_ba) / args.epochs))
    # print('FedCSL算法 Average Local F1-score: {:.4f}'.format(sum(fedavg_local_test_f1_score) / args.epochs))
    # print('FedCSL算法 Average Local Recall: {:.4f}'.format(sum(fedavg_local_test_recall) / args.epochs))
    avg_epoch_time = sum(avg_epoch_time) / len(avg_epoch_time)
    fedavg_train_loss = [round(i, 4) for i in fedavg_train_loss]
    fedavg_test_ba = [round(i, 4) for i in fedavg_test_ba]
    fedavg_test_gmean = [round(i, 4) for i in fedavg_test_gmean]
    fedavg_test_f1_score = [round(i, 4) for i in fedavg_test_f1_score]
    fedavg_test_recall = [round(i, 4) for i in fedavg_test_recall]
    print('FedCSL-CM算法 Average Global AUC-ROC: {:.4f}'.format(sum(fedavg_test_gmean) / args.epochs))
    print('FedCSL-CM算法 Average Global AUC-PR: {:.4f}'.format(sum(fedavg_test_ba) / args.epochs))
    print('FedCSL-CM算法 Average Global BS+: {:.4f}'.format(sum(fedavg_test_f1_score) / args.epochs))
    print('FedCSL-CM算法 Average Global KS: {:.4f}'.format(sum(fedavg_test_recall) / args.epochs))
    # print('FedCSL-CM算法 Average Global G-mean: {:.4f}'.format(sum(fedavg_test_gmean) / args.epochs))
    # print('FedCSL-CM算法 Average Global BA: {:.4f}'.format(sum(fedavg_test_ba) / args.epochs))
    # print('FedCSL-CM算法 Average Global F1-score: {:.4f}'.format(sum(fedavg_test_f1_score) / args.epochs))
    # print('FedCSL-CM算法 Average Global Recall: {:.4f}'.format(sum(fedavg_test_recall) / args.epochs))
    # print('FedCSL-CM算法 Average Epoch Time: {:.4f}s'.format(avg_epoch_time))
    fedavg_train_loss.insert(0, 'Train loss')
    fedavg_test_gmean.insert(0, 'AUC-ROC')
    fedavg_test_ba.insert(0, 'AUC-PR')
    fedavg_test_f1_score.insert(0, 'BS+')
    fedavg_test_recall.insert(0, 'KS')
    FedCSL_result = [fedavg_train_loss, fedavg_test_gmean, fedavg_test_ba, fedavg_test_f1_score, fedavg_test_recall]
    list2txt(FedCSL_result, "./save/{} FedCSL_CM results.txt".format(args.dataset))