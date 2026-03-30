from scipy.stats import friedmanchisquare
from scipy.stats import f
import matplotlib.pyplot as plt
import Orange
# import scikit_posthocs as sp
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(-matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n + 1
            ## 相同排名评分序值
            if j < 3 and matrix[i, sorts[i, j]] == matrix[i, sorts[i, j - 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i, j]] != matrix[i, sorts[i, j - 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i, sorts[i, j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0

            else:
                matrix[i, sorts[i, j]] = j + 1
                continue
    matrix=np.sum(matrix, axis=0) / 3   # 返回矩阵每一列的平均值
    return matrix
# 临界值查询
m = 5   # 13                                       # m = 算法个数K-1
n = 25   # 39                                       # n = (数据集个数n-1)*(k-1)
alpha=0.05                                        # 设置alpha
fc=f.isf(q=alpha, dfn=m, dfd=n)
print('临界值：',fc)

names = ['FedAvg','FedProx','SCAFFOLD','FedNova','FedKD','FedCSL-CM']
A_AUCROC = [0.8764,	0.8764,	0.8785,	0.8785,	0.8753,	0.8807]
LD_AUCROC = [ 0.5569,	0.5586,	0.5609,	0.5656,	0.550,	0.6043]
HMEQ_AUCROC = [0.7387,	0.7387,	0.7414,	0.7520,	0.736,	0.7636]
Taiwan_AUCROC = [0.7762,	0.7761,	0.7735,	0.7740,	0.7754,	0.7796]
LC_AUCROC = [0.7903,	0.7891,	0.7922,	0.8086,	0.7783,	0.8489]
GMSC_AUCROC = [0.6343,	0.6342,	0.6343,	0.6219,	0.6334,	0.6401]


matrix_aucroc =np.array([LD_AUCROC,HMEQ_AUCROC,Taiwan_AUCROC,GMSC_AUCROC,A_AUCROC,LC_AUCROC])
matrix_r1 = rank_matrix(matrix_aucroc)
stat, p = friedmanchisquare(LD_AUCROC,HMEQ_AUCROC,Taiwan_AUCROC,GMSC_AUCROC,A_AUCROC,LC_AUCROC)
datasets_num = 6
avranks = matrix_r1

CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
print(CD)
# CD = 2.77
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/FL_CD_AUC-ROC.png",dpi=3600, bbox_inches='tight')
print('AUC-ROC')
for i in matrix_r1:
    print(np.round(i,2))
print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
A_AUCPR = [0.8793,	0.8793,	0.8807,	0.8795,	0.8768,	0.8707]
LD_AUCPR = [0.3385,	0.3391,	0.3382,	0.3581	,0.346,	0.3755]
HMEQ_AUCPR = [0.4394,	0.4392,	0.4414,	0.4515,	0.435,	0.4658]
Taiwan_AUCPR = [0.5577,	0.5576	,0.5575	,0.5616,	0.553,	0.5677]
LC_AUCPR = [0.5285,	0.5261,	0.5333,	0.5904,	0.5022,	0.6647]
GMSC_AUCPR = [0.1059,	0.1057,	0.1059,	0.1068,	0.105,	0.1170]

matrix_aucpr =np.array([LD_AUCPR, HMEQ_AUCPR,Taiwan_AUCPR,GMSC_AUCPR,A_AUCPR ,LC_AUCPR ])
matrix_r2 = rank_matrix(matrix_aucpr)
stat, p = friedmanchisquare(LD_AUCPR, HMEQ_AUCPR,Taiwan_AUCPR,GMSC_AUCPR,A_AUCPR ,LC_AUCPR)
avranks = matrix_r2

# CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/FL_CD_AUC-PR.png",dpi=3600, bbox_inches='tight')
print('AUC-PR')
for i in matrix_r2:
    print(np.round(i,2))
print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
A_KS = [0.7089,	0.7089,	0.7232,	0.7114,	0.7050,	0.7203]
LD_KS = [0.2667,	0.267,	0.2681,	0.2918,	0.280,	0.3338]
HMEQ_KS = [0.4558,	0.4570,	0.4557,	0.4594,	0.453,	0.4915]
Taiwan_KS = [0.4458,	0.4455,	0.4433,	0.4485,	0.4449,	0.4521]
LC_KS = [0.4785,	0.4759,	0.4823,	0.4963,	0.4567,	0.5586]
GMSC_KS = [0.2289,	0.2287,	0.2291,	0.2114,	0.229,	0.2337]



matrix_KS = np.array([LD_KS,HMEQ_KS,Taiwan_KS,GMSC_KS,A_KS,LC_KS])
matrix_r3 = rank_matrix(matrix_KS)
stat, p = friedmanchisquare(LD_KS,HMEQ_KS,Taiwan_KS,GMSC_KS,A_KS,LC_KS)
avranks = matrix_r3

# CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/FL_CD_KS.png",dpi=3600, bbox_inches='tight')
print('KS')
for i in matrix_r3:
    print(np.round(i,2))
print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
A_BS = [0.2497 ,	0.2498,	0.2505,	0.2142,	0.2525,	0.0423]
LD_BS = [0.5312,	0.5307,	0.5428,	0.5351	,0.531,	0.1747]
HMEQ_BS = [	0.5870	,0.5876,	0.5880,	0.5183,	0.597,	0.2418]
Taiwan_BS = [0.4305	,0.4302	,0.4375,	0.4279	,0.434,	0.2111]
LC_BS = [0.6013,	0.6021,	0.5997,	0.4423,	0.6071,	0.1837]
GMSC_BS = [0.8641,	0.8638,	0.8664,	0.8317,	0.863,	0.5979]

matrix_BS = -np.array([LD_BS,HMEQ_BS,Taiwan_BS,GMSC_BS,A_BS,LC_BS])
matrix_r4 = rank_matrix(matrix_BS)
stat, p = friedmanchisquare(LD_BS,HMEQ_BS,Taiwan_BS,GMSC_BS,A_BS,LC_BS)
avranks = matrix_r4

# CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/FL_CD_BS+.png",dpi=3600, bbox_inches='tight')
print('BS+')
for i in matrix_r4:
    print(np.round(i,2))
print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')
Avgrank = matrix_r1+matrix_r2+matrix_r3+matrix_r4
print(Avgrank/4)
for i in Avgrank/4:
    print(np.round(i,2))



