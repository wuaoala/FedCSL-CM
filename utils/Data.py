
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import re
torch.set_default_tensor_type(torch.FloatTensor)
'''  标准差归一化（Z-score）
numvars = numvars.apply(lambda x: (x - x.mean()) / x.std())
'''
def identify_columns(df, threshold=25):
    # Identify categorical object columns, categorical numerical columns, and non-categorical columns
    cat_object_list = [i for i in df.columns if df[i].dtype == 'object']
    cat_num_list = [i for i in df.columns if df[i].dtype in ['int64', 'float64'] and df[i].nunique() < threshold]
    catvars_list = cat_object_list + cat_num_list
    # cat_list = [i for i in df.columns if df[i].nunique() < threshold]
    non_cat_list = [i for i in df.columns if i not in cat_object_list and i not in cat_num_list]
    # Identify object columns and numerical columns in non-categorical columns
    mix_serise_col = df[non_cat_list]
    non_cat_obj = [i for i in mix_serise_col.columns if mix_serise_col[i].dtype == 'object']
    non_cat_num = [i for i in mix_serise_col.columns if mix_serise_col[i].dtype in ['int64', 'float64']]
    numvars_list = non_cat_num
    # #Print the results
    # print('Categorical object columns:', len(cat_object_list))
    # print('Categorical numerical columns:', len(cat_num_list))
    # print('Non-categorical columns:', len(non_cat_list))
    # print('Object columns in non-categorical columns:', len(non_cat_obj))
    # print('Numerical columns in non-categorical columns:', len(non_cat_num))

    # print('类别变量个数', len(catvars_list))
    # print('连续变量个数', len(numvars_list))
    # Return the results as a dictionary
    results = {
        'catvars_list': catvars_list,
        'numvars_list': numvars_list}
    # results = {
    #     'cat_object_list': cat_object_list,
    #     'cat_num_list': cat_num_list,
    #     'cat_list': cat_list,
    #     'non_cat_list': non_cat_list,
    #     'non_cat_obj': non_cat_obj,
    #     'non_cat_num': non_cat_num
    # }
    return results
def label_encoding(df):
    le_vars = []
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(df[col].unique()) == 2:
                le_vars.append(col)
                le = LabelEncoder()
                le.fit(df[col])
                df[col] = le.transform(df[col])
            # else:
            #     df[col] = pd.get_dummies(df[col])
    # print(df[le_vars])
    return df[le_vars]
def missing_values_table(df):
    # Check if input is a dataframe or a series
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    # Get columns with missing values
    na_columns = df.columns[df.isnull().any()].tolist()
    # Count missing values and calculate ratio
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / df.shape[0] * 100).sort_values(ascending=False)
    # Create DataFrame with missing values and ratio
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Missing Values', 'Percentage'])
    print(missing_df)
    return missing_df
def missing_preprocess_data(df):
    # Identify columns with more than 60% missing values
    missing_cols = df.columns[df.isnull().mean() > 0.6]
    # Drop columns with more than 60% missing values
    num_cols_dropped = len(missing_cols)
    df.drop(columns=missing_cols, inplace=True)
    print(f'Dropped {num_cols_dropped} columns due to missing value threshold')
    results = identify_columns(df)
    numvars_list = results['numvars_list']
    catvars_list = results['catvars_list']
    # Fill remaining missing values using median imputation
    # imputer = SimpleImputer(strategy='median')
    # df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df[numvars_list] = df[numvars_list].fillna(df[numvars_list].mean())
    for col in catvars_list:
        df[col] = df[col].fillna(df[col].value_counts().index[0])
    # Print out the original and preprocessed datasets
    # print('Preprocessed dataset:')
    # print(df)
    return df

def HomeCredit():
    data = pd.read_csv('./creditdatasets/application_train.csv', delimiter=',')
    data.rename(columns={'TARGET': 'target'}, inplace=True)
    print(data['target'].value_counts())
    train_dataset = data
    all_features = list(train_dataset.columns)
    all_features.pop(0)
    all_features.pop(0)
    train_fea = train_dataset.loc[:,all_features]
    train_fea = missing_preprocess_data(train_fea)
    levars = label_encoding(train_fea)
    results = identify_columns(train_fea)
    # cat_object_list = results['cat_object_list']
    # cat_num_list = results['cat_num_list']
    catvars_list = results['catvars_list']
    numvars_list = results['numvars_list']
    # print(len(catvars_list))
    # print(len(numvars_list))
    dummyvars = pd.get_dummies(train_fea[catvars_list],columns=catvars_list)
    numvars = train_fea[numvars_list]
    # numvars = numvars.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    numvars = numvars.apply(lambda x: (x - x.mean()) / x.std())
    train_fea = pd.concat([numvars, levars, dummyvars], axis=1)
    train_labels = train_dataset.loc[:,'target']
    train_dataset = pd.concat([train_fea, train_labels], axis=1)
    # print(train_dataset)
    return train_dataset, train_dataset
# HomeCredit()
def Hmeq():
    data = pd.read_csv('./creditdatasets/hmeq.csv', delimiter=',')
    data.rename(columns={'BAD': 'target'}, inplace=True)
    print(data['target'].value_counts())
    train_dataset = data
    all_features = list(train_dataset.columns)
    all_features.pop(0)
    train_fea = train_dataset.loc[:,all_features]
    train_fea = missing_preprocess_data(train_fea)
    levars = label_encoding(train_fea)
    results = identify_columns(train_fea)
    catvars_list = results['catvars_list']
    numvars_list = results['numvars_list']
    dummyvars = pd.get_dummies(train_fea[catvars_list], columns=catvars_list, dtype=float)
    numvars = train_fea[numvars_list]
    # numvars = numvars.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    numvars = numvars.apply(lambda x: (x - x.mean()) / x.std())

    train_fea = pd.concat([numvars, levars, dummyvars], axis=1)
    train_labels = train_dataset.loc[:,'target']
    train_dataset = pd.concat([train_fea, train_labels], axis=1)
    # print(train_dataset)
    return train_dataset, train_dataset
# Hmeq()
def Taiwan():
    data = pd.read_csv('./creditdatasets/Taiwan.csv',header=1)
    data.rename(columns={'default payment next month': 'target'}, inplace=True)
    # data.target.replace([1, 0], [0, 1], inplace=True)
    train_dataset = data
    print(data['target'].value_counts())
    all_features = list(train_dataset.columns)
    all_features.pop(0)
    all_features.pop(-1)
    train_fea = train_dataset.loc[:,all_features]
    numvars = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
               'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
               'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    catvars = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    numeric_features = numvars
    # train_fea.loc[:,numeric_features] = train_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    train_fea.loc[:, numeric_features] = train_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    train_fea = train_fea.astype('str')
    dummyvars = pd.get_dummies(train_fea[catvars])
    train_fea = pd.concat([train_fea[numeric_features], dummyvars], axis=1)
    train_fea = train_fea.astype('float')
    train_labels = train_dataset.loc[:,'target']
    train_dataset = pd.concat([train_fea,train_labels],axis=1)
    # print(train_dataset)
    return train_dataset,train_dataset

def Give_me_some_credit():
    data = pd.read_csv('./creditdatasets/cs-training.csv')
    data.rename(columns={'SeriousDlqin2yrs':'target'}, inplace=True)
    print(data['target'].value_counts())
    train_dataset = data
    all_features = list(train_dataset.columns)
    all_features.pop(0)
    all_features.pop(0)
    train_fea = train_dataset.loc[:,all_features]
    train_fea = missing_preprocess_data(train_fea)
    numeric_features = list(train_fea.select_dtypes(exclude=['object']).columns)
    train_fea.loc[:,numeric_features] = train_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    # train_fea.loc[:, numeric_features] = train_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    train_labels = train_dataset.loc[:,'target']
    train_dataset = pd.concat([train_fea, train_labels], axis=1)
    # print(train_dataset)
    return train_dataset,train_dataset
# Give_me_some_credit()
def Loan_Data():
    data = pd.read_csv('./creditdatasets/Loan Data.csv',delimiter=';')
    data.rename(columns={'BAD':'target'}, inplace=True)
    # data.target.replace([1, 0], [0, 1], inplace=True)
    train_dataset = data
    print(data['target'].value_counts())
    all_features = list(train_dataset.columns)
    all_features.pop(-1)
    train_fea = train_dataset.loc[:,all_features]
    catvars = ['AES','RES']
    numvars = [k for k in all_features if k not in catvars]
    numeric_features = numvars
    # train_fea.loc[:,numeric_features] = train_fea.loc[:,numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    train_fea.loc[:, numeric_features] = train_fea.loc[:, numeric_features].apply(lambda x: (x - x.mean()) / x.std())
    dummyvars = pd.get_dummies(train_fea[catvars], dtype=float)
    train_fea = pd.concat([train_fea[numeric_features], dummyvars], axis=1)
    train_labels = train_dataset.loc[:,'target'].astype('int')
    train_dataset = pd.concat([train_fea, train_labels], axis=1)
    # print(train_dataset)
    return train_dataset, train_dataset
# Loan_Data()

def Australian():
    names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
             'A7','A8', 'A9', 'A10','A11', 'A12', 'A13', 'A14', 'target']
    # data = pd.read_csv('E:\Software\PycharmProjects\pythonProject\FedCSL-CM\creditdatasets\Australian.csv',names=names,
    #                    delimiter=',')
    data = pd.read_csv('./creditdatasets/Australian.csv', names=names, delimiter=',')
    print(data['target'].value_counts())
    train_dataset = data
    all_features = list(train_dataset.columns)
    all_features.pop(-1)
    train_fea = train_dataset[all_features]
    numvars = ['A2', 'A3', 'A7', 'A13', 'A14', 'A10']
    catvars = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9','A11', 'A12']
    numeric_features = numvars
    train_fea.loc[:, numeric_features] = train_fea.loc[:, numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    # train_fea[numeric_features] = train_fea[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    train_fea = train_fea.astype('str')
    dummyvars = pd.get_dummies(train_fea[catvars])
    train_fea = pd.concat([train_fea[numeric_features], dummyvars], axis=1)
    train_fea = train_fea.astype('float')
    train_labels = train_dataset['target']
    train_dataset = pd.concat([train_fea, train_labels], axis=1)
    # print(train_dataset)
    return train_dataset, train_dataset
# Australian()
def Lendingclub():
    data = pd.read_csv('./creditdatasets/lending club2005_2012.csv', delimiter=',')
    # data = pd.read_csv('E:\Software\PycharmProjects\pythonProject\IVLR-ACS\data\lending club2005_2012.csv', delimiter=',')
    data.rename(columns={'lable': 'target'}, inplace=True)
    data = data.sample(frac=0.8, random_state=11)
    data.columns = [
        re.sub(r'[^A-Za-z0-9_]+', '_', col)  # 将非法字符替换为下划线
        for col in data.columns]

    print(data['target'].value_counts())
    all_features = list(data.columns)
    all_features.pop(-1)
    train_fea = data.loc[:, all_features]
    numvars = all_features[:30]
    catvars = all_features[30:]
    numeric_features = numvars
    train_fea.loc[:, numeric_features] = train_fea.loc[:, numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
    train_fea = pd.concat([train_fea[numeric_features], train_fea[catvars]], axis=1)
    train_fea = missing_preprocess_data(train_fea)
    train_labels = data.loc[:, 'target']
    train_dataset = pd.concat([train_fea, train_labels], axis=1)
    # print(train_dataset)
    return train_dataset, train_dataset
# Lendingclub()

def get_dataset(name):
    if name == 'Taiwan':
        dataset_train , dataset_test = Taiwan()
    elif name == 'HMEQ':
        dataset_train , dataset_test = Hmeq()
    elif name == 'GMSC':
        dataset_train , dataset_test = Give_me_some_credit()
    elif name == 'Loan Data':
        dataset_train , dataset_test = Loan_Data()
    elif name == 'HC':
        dataset_train , dataset_test = HomeCredit()
    elif name == 'A':
        dataset_train , dataset_test = Australian()
    elif name == 'LC':
        dataset_train , dataset_test = Lendingclub()
    else:
        exit('Error: unrecognized dataset')
    return dataset_train , dataset_test


