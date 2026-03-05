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
alpha = 0.05                                        # 设置alpha
fc = f.isf(q=alpha, dfn=m, dfd=n)
print('临界值：',fc)
names = ['LR','SVM','XGB','LightGBM','MLP','FedCSL-CM']
LD_AUCROC = [0.5794,	0.5840,	0.5568,	0.5627,	0.5717,	0.6043]
HMEQ_AUCROC = [0.7365,	0.7339	,0.7433,	0.7394,	0.7281,	0.7636]
Taiwan_AUCROC = [0.7662	,0.7102,	0.7586,	0.7508,	0.7353,	0.7796]
GMSC_AUCROC = [0.6463,	0.6465,	0.6546,	0.6546,	0.6477	,0.6401]
A_AUCROC = [0.8513,	0.8536	,0.8499	,0.8305,	0.8459,	0.8807]
LC_AUCROC = [0.7437,	0.7731,	0.7757	,0.7705	,0.7589,	0.8489]

matrix_aucroc =np.array([LD_AUCROC,HMEQ_AUCROC,Taiwan_AUCROC,GMSC_AUCROC,A_AUCROC,LC_AUCROC])
matrix_r1 = rank_matrix(matrix_aucroc)
stat, p = friedmanchisquare(LD_AUCROC,HMEQ_AUCROC,Taiwan_AUCROC,GMSC_AUCROC,A_AUCROC,LC_AUCROC)
datasets_num = 6
avranks = matrix_r1

CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/L_CD_AUCROC.png",dpi=3600)
# print(CD)
# print('AUC-ROC',matrix_r1)
print('AUC-ROC')
for i in matrix_r1:
    print(np.round(i,2))
print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

LD_AUCPR = [0.3281,	0.3303	,0.3418,	0.3928,	0.3048,	0.3755]
HMEQ_AUCPR = [0.4875	,0.4624,	0.4924,	0.4681,	0.4881,	0.4658]
Taiwan_AUCPR = [0.5422	,0.5122	,0.5387	,0.5357	,0.4902	,0.5677]
GMSC_AUCPR = [0.1115	,0.1106	,0.1160	,0.1160	,0.1132	,0.1170]
A_AUCPR = [0.8515,	0.8542,	0.8560	,0.8522,	0.8271,	0.8707]
LC_AUCPR = [0.4767,	0.4425	,0.4387	,0.4251,	0.5092,	0.6647]

matrix_aucpr =np.array([LD_AUCPR, HMEQ_AUCPR,Taiwan_AUCPR,GMSC_AUCPR,A_AUCPR ,LC_AUCPR ])
matrix_r2 = rank_matrix(matrix_aucpr)
stat, p = friedmanchisquare(LD_AUCPR, HMEQ_AUCPR,Taiwan_AUCPR,GMSC_AUCPR,A_AUCPR ,LC_AUCPR)
avranks = matrix_r2

CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/L_CD_AUCPR.png",dpi=3600)
# print('AUC-PR',matrix_r2)
print('AUC-PR')
for i in matrix_r2:
    print(np.round(i,2))

print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

LD_KS= [0.2867,	0.2928,	0.2165,	0.1874,	0.2830,	0.3338]
HMEQ_KS = [0.4266,	0.4294,	0.4324,	0.4268,	0.4261,	0.4915]
Taiwan_KS = [0.4283,	0.3782,	0.4255,	0.4146,	0.3783,	0.4521]
GMSC_KS = [0.2205,	0.2204	,0.2093,	0.2093,	0.2341,	0.2337]
A_KS = [0.6849,	0.7133,	0.6590,	0.6590,	0.6760,	0.7203]
LC_KS = [0.4605,	0.4580,	0.5219,	0.5129,	0.5055,	0.5586]

matrix_KS = np.array([LD_KS,HMEQ_KS,Taiwan_KS,GMSC_KS,A_KS,LC_KS])
matrix_r3 = rank_matrix(matrix_KS)
stat, p = friedmanchisquare(LD_KS,HMEQ_KS,Taiwan_KS,GMSC_KS,A_KS,LC_KS)
avranks = matrix_r3

CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/L_CD_KS.png",dpi=3600)
# print('KS',matrix_r3)
print('KS')
for i in matrix_r3:
    print(np.round(i,2))
print('stat= %.3f' % stat,  'p=',np.array(p))
if p > 0.05:
	print('Probably the same distribution')
else:
	print('Probably different distributions')

LD_BS = [0.5336,	0.5282,	0.4189,	0.5262,	0.5814,	0.1747]
HMEQ_BS = [0.4906,	0.5307,	0.4368,	0.6020,	0.4817,	0.2418]
Taiwan_BS = [0.4370,	0.4859,	0.4714,	0.4973,	0.4500,	0.2111]
GMSC_BS = [0.8407,	0.8398,	0.8198	,0.8445,	0.8490,	0.5979]
A_BS = [0.1600,	0.1542,	0.1709,	0.2177,	0.1976,	0.0423]
LC_BS = [0.5659,	0.5400,	0.5361,	0.5779,	0.6020,	0.1837]

matrix_BS = -np.array([LD_BS,HMEQ_BS,Taiwan_BS,GMSC_BS,A_BS,LC_BS])
matrix_r4 = rank_matrix(matrix_BS)
stat, p = friedmanchisquare(LD_BS,HMEQ_BS,Taiwan_BS,GMSC_BS,A_BS,LC_BS)
avranks = matrix_r4

CD = Orange.evaluation.scoring.compute_CD(avranks, datasets_num, alpha='0.05', test='nemenyi')
Orange.evaluation.scoring.graph_ranks(avranks, names, cd=CD, width=4, textspace=1, reverse=True)
plt.savefig("./save/L_CD_BS.png",dpi=3600)
# print('BS+',matrix_r4)
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



