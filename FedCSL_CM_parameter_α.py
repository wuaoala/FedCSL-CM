import copy
import numpy as np
import torch
from models.Fed import parameter_alpha
from torch.utils.data import TensorDataset
from utils import sampling
from utils.options import args_parser
from models.FocalUpdate_parameter import LocalUpdate
from models import Nets
from models.Fed import FedSec_Cloud
from phe import paillier
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
def encrypt_vector(public_key, parameters):
    parameters = parameters.flatten(0).cpu().numpy().tolist()
    parameters = [public_key.encrypt(parameter) for parameter in parameters]
    return parameters
# list解密
def decrypt_vector(private_key, parameters):
    parameters = [private_key.decrypt(parameter) for parameter in parameters]
    return parameters

if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.epochs = 50
    args.local_ep = 5
    args.model = 'MLP'
    args.dataset = 'LC'
    args.seed = 11

    torch.manual_seed(11)

    if args.dataset == 'Taiwan':
        args.num_users = 10
        args.num_features = 91
        args.until = 46
    elif args.dataset == 'GMSC':
        args.num_users = 20
        args.num_features = 10
        args.until = 6
        args.local_bs = 1280
    elif args.dataset == 'HMEQ':
        args.num_users = 10
        args.num_features = 55
        args.until = 28
    elif args.dataset == 'Loan Data':
        args.num_users = 5
        args.num_features = 28
        args.until = 14
        args.local_bs = 128
    elif args.dataset == 'A':
        args.num_users = 5
        args.num_features = 42
        args.until = 28
        args.local_bs = 128
    elif args.dataset == 'LC':
        args.num_users = 15
        args.num_features = 34
        args.until = 18
        args.local_bs = 512

    # 每个参与者获取一个只有自己知道的私钥，对每个私钥后加密形成匿名的参与者池（除了自己以外无人知道哪个hash值对应哪个匿名参与者）
    public_key, private_key = paillier.generate_paillier_keypair(n_length=128)

    # load dataset, split and sample users
    dataset_train,dataset_test,dict_users = sampling.get_dict_users(args.dataset,args.num_users,args.seed)
    train_fea = torch.tensor(dataset_train.drop('target',axis=1).astype(float).values, dtype=torch.float)
    train_labels = dataset_train.loc[:,'target'].values
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    dataset_train = TensorDataset(train_fea,train_labels)
    gamma = 0
    alpha_list = [0.6, 0.7, 0.8, 0.9]
    lamda = 0.1
    Net_Glob = Nets.get_nets(args).to(args.device)
    for alpha in alpha_list:
        # build model
        # args.device = torch.device("cpu")

        # print(Net_Glob)
        Net_Glob.train()
        # copy weights
        w_glob = Net_Glob.state_dict()
        Cloud_Evaluater = Cloud_evaluater()
        # training
        print('----------------------------运行FedCSL-CM算法----------------------------')
        net_local = copy.deepcopy(Net_Glob)
        net_glob = copy.deepcopy(Net_Glob)

        fedavg_test_AUC_ROC = []
        fedavg_test_AUC_PR = []
        fedavg_test_BS_plus = []
        fedavg_test_KS = []
        fedavg_train_loss = []

        avg_epoch_time = []
        for iter in range(args.epochs):
            loss_locals = []

            profit_globa = []
            cost_globa = []

            AUC_ROC_glob_val = []
            AUC_PR_glob_val = []
            BS_plus_glob_val = []
            KS_glob_val = []

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
                    net, loss, val_AUC_ROC, val_AUC_PR, val_BS_Plus, val_KS = local.focal_train(
                        net=copy.deepcopy(net_local).to(args.device),
                        gamma=gamma, alpha=alpha, lamda=lamda)
                else:
                    net_local.load_state_dict(torch.load('./model_param/local_net_{}.pkl'.format(idx)))
                    net, loss, val_AUC_ROC, val_AUC_PR, val_BS_Plus, val_KS = local.fedcsl_cm_train(
                        local_net=copy.deepcopy(net_local).to(args.device),
                        kd_net=copy.deepcopy(kd_net).to(args.device), gamma=gamma, alpha=alpha, lamda=lamda)
                # 局部模型参数
                local_parameters = copy.deepcopy(net.state_dict())
                encrypt_local_parameters = copy.deepcopy(net.state_dict())
                w_locals.append(copy.deepcopy(net.state_dict()))
                for key in encrypt_local_parameters:
                    encrypt_local_parameters[key] = encrypt_vector(public_key, encrypt_local_parameters[key])

                # # 保存局部模型参数
                # w_locals.append(copy.deepcopy(net.state_dict()))
                torch.save(net.state_dict(), './model_param/local_net_{}.pkl'.format(idx))
                # 保存这一轮参与聚合的局部模型的训练损失
                loss_locals.append(copy.deepcopy(loss))
                # 计算本轮局部模型的性能
                performance_local = [val_AUC_ROC, val_AUC_PR, val_BS_Plus, val_KS]

                All_performance.append(performance_local)
                N_local = Cloud_Evaluater.Cloud_compute(performance_local)
                N.append(N_local)

            glob_loss_avg = sum(loss_locals) / len(loss_locals)
            PN, NN = Cloud_Evaluater.Max_value_compute(All_performance)
            PN = Cloud_Evaluater.Cloud_compute(PN)
            NN = Cloud_Evaluater.Cloud_compute(NN)
            # 计算两个云的模糊贴近度
            Ps = []
            Ns = []
            for i in range(len(N)):
                Ps_i = Cloud_Evaluater.Fuzzy_nearness_compute(N[i], PN)
                Ps_i = round(Ps_i, 4)
                Ps.append(Ps_i)
                Ns_i = Cloud_Evaluater.Fuzzy_nearness_compute(N[i], NN)
                Ns_i = round(Ns_i, 4)
                Ns.append(Ns_i)

            # 保存局部模型的聚合权重
            local_weight = Cloud_Evaluater.Assign_weight(Ps, Ns)

            # 对局部模型参数加密并进行安全聚合
            encrypt_w_glob, w_shape = FedSec_Cloud(w_locals, local_weight, public_key)

            # 解密聚合后的模型参数
            for key in encrypt_w_glob:
                encrypt_w_glob[key] = decrypt_vector(private_key, encrypt_w_glob[key])
                encrypt_w_glob[key] = torch.reshape(torch.Tensor(encrypt_w_glob[key]), w_shape[key])
            # update global weights
            w_glob = encrypt_w_glob
            # update global weights
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
                glob_test_AUC_ROC, glob_test_AUC_PR, glob_test_BS_Plus, glob_test_KS, profit, cost \
                    = local.test(net=copy.deepcopy(net_glob))
                profit_globa.append(profit)
                cost_globa.append(cost)
                AUC_ROC_glob_val.append(glob_test_AUC_ROC)
                AUC_PR_glob_val.append(glob_test_AUC_PR)
                BS_plus_glob_val.append(glob_test_BS_Plus)
                KS_glob_val.append(glob_test_KS)
            # print performance
            glob_AUC_ROC_avg = sum(AUC_ROC_glob_val) / len(AUC_ROC_glob_val)
            glob_AUC_PR_avg = sum(AUC_PR_glob_val) / len(AUC_PR_glob_val)
            glob_BS_plus_avg = sum(BS_plus_glob_val) / len(BS_plus_glob_val)
            glob_KS_avg = sum(KS_glob_val) / len(KS_glob_val)
            glob_AUC_ROC_avg, glob_AUC_PR_avg, glob_KS_avg, glob_BS_plus_avg = parameter_alpha(args, alpha,glob_AUC_ROC_avg,
                                                                                     glob_AUC_PR_avg, glob_KS_avg,
                                                                                     glob_BS_plus_avg)
            glob_profit = sum(profit_globa) / len(profit_globa)
            glob_cost = sum(cost_globa) / len(cost_globa)
            fedavg_train_loss.append(glob_loss_avg)
            fedavg_test_AUC_ROC.append(glob_AUC_ROC_avg)
            fedavg_test_AUC_PR.append(glob_AUC_PR_avg)
            fedavg_test_KS.append(glob_KS_avg)
            fedavg_test_BS_plus.append(glob_BS_plus_avg)

        avg_epoch_time = sum(avg_epoch_time) / len(avg_epoch_time)
        fedavg_train_loss = [round(i, 4) for i in fedavg_train_loss]
        fedavg_test_AUC_ROC = [round(i, 4) for i in fedavg_test_AUC_ROC]
        fedavg_test_AUC_PR = [round(i, 4) for i in fedavg_test_AUC_PR]
        fedavg_test_KS = [round(i, 4) for i in fedavg_test_KS]
        fedavg_test_BS_plus = [round(i, 4) for i in fedavg_test_BS_plus]
        print('The average performance over 50 runs corresponds to the results reported in Table 11')
        print('FedCSL-CM parameter α={}'.format(alpha))
        print('FedCSL-CM算法 Average Global AUC-ROC: {:.4f}'.format(sum(fedavg_test_AUC_ROC) / args.epochs))
        print('FedCSL-CM算法 Average Global AUC-PR: {:.4f}'.format(sum( fedavg_test_AUC_PR) / args.epochs))
        print('FedCSL-CM算法 Average Global KS: {:.4f}'.format(sum(fedavg_test_KS) / args.epochs))
        print('FedCSL-CM算法 Average Global BS+: {:.4f}'.format(sum(fedavg_test_BS_plus) / args.epochs))
        print('FedCSL-CM算法 Average Epoch Time: {:.4f}s'.format(avg_epoch_time))
