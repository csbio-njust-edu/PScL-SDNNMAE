import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def path_in(label_dic, read_path, write_path):

    bio_set = os.listdir(read_path)

    total = 0
    for subcell in bio_set:
        sub_path1 = os.path.join(read_path, subcell)
        sub_set1 = os.listdir(sub_path1)
        for sub_num1 in sub_set1:
            sub_path2 = os.path.join(sub_path1, sub_num1)
            sub_set2 = os.listdir(sub_path2)
            for sub_num2 in sub_set2:
                sub_path3 = os.path.join(sub_path2, sub_num2)
                patches = os.listdir(sub_path3)
                for filename in patches:
                    if filename.endswith('.jpg'):
                        total = total + 1

    data = pd.DataFrame(index=np.arange(0, total)
                        , columns=['path', 'label'])

    idx = 0
    for subcell in bio_set:
        sub_path1 = os.path.join(read_path, subcell)
        sub_set1 = os.listdir(sub_path1)
        for sub_num1 in sub_set1:
            sub_path2 = os.path.join(sub_path1, sub_num1)
            sub_set2 = os.listdir(sub_path2)
            for sub_num2 in sub_set2:
                sub_path3 = os.path.join(sub_path2, sub_num2)
                patches = os.listdir(sub_path3)
                for filename in patches:
                    if filename.endswith('.jpg'):
                        data.loc[idx, 'path'] = os.path.join(sub_path3, filename)
                        data.loc[idx, 'label'] = label_dic.get(subcell) - 1
                        idx += 1

    if write_path:
        data.to_csv(write_path, index=False)
    else:
        return data

def split_dataset(df, split_rate=0.8):
    # Empty DataFrames for training and validation sets
    train_df = pd.DataFrame(columns=['path', 'label'])
    val_df = pd.DataFrame(columns=['path', 'label'])

    # Group by label
    grouped = df.groupby('label')

    # For each label, split the data according to the split rate
    for _, group in grouped:
        train, val = train_test_split(group, train_size=split_rate, shuffle=True, random_state=420)
        train_df = pd.concat([train_df, train])
        val_df = pd.concat([val_df, val])

    return train_df, val_df

def split_train_test(dataframe, k):
    # 验证输入
    if 'label' not in dataframe.columns:
        raise ValueError("dataframe must have a 'label' column")

    # 初始化StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # 分层抽样切分数据
    X = dataframe.drop(columns=['label'])  # 特征
    y = dataframe['label']  # 标签

    train_sets = []  # 存储训练集的列表
    test_sets = []  # 存储测试集的列表

    # 切分数据
    for train_index, test_index in skf.split(X, y):
        # 根据索引切分数据
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 将训练集和测试集合并回DataFrame
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # 添加到列表
        train_sets.append(train_df)
        test_sets.append(test_df)

    return train_sets, test_sets


def stratified_k_fold_split(dataframe, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=420)

    train_list = []
    val_list = []

    if not pd.api.types.is_numeric_dtype(dataframe['label']):
        le = LabelEncoder()
        dataframe['label'] = le.fit_transform(dataframe['label'])

    X = dataframe.index.to_numpy()
    y = dataframe['label'].to_numpy()

    for train_index, val_index in skf.split(X, y):
        train_list.append(dataframe.iloc[train_index])
        val_list.append(dataframe.iloc[val_index])

    return train_list, val_list

if __name__ == '__main__':
    if os.path.exists("./data_path") is False:
        os.makedirs("./data_path")
    Path = "../data/dataset/"
    label_dic = {
        '2384': 1,
        '4125': 1,
        '29804': 1,
        '2321': 1,
        '3901': 2,
        '2645': 2,
        '7912': 2,
        '35593': 3,
        '11929': 3,
        '35866': 3,
        '6964': 4,
        '1467': 4,
        '41690': 5,
        '4480': 5,
        '26533': 5,
        '53891': 6,
        '54422': 6,
        '28136': 6,
        '43912': 6,
        '17964': 7,
        '23099': 7,
        '19025': 7,
        '22012': 7
    }

    data = path_in(label_dic=label_dic, read_path=Path, write_path=None)
    K = 10
    train_data_list, val_data_list = stratified_k_fold_split(dataframe=data, k=K)
    for i in range(K):
        write_dir = "./data_path/path" + str(i)
        if os.path.exists(write_dir) is False:
            os.makedirs(write_dir)
        write_path_train = write_dir + "/train.csv"
        write_path_val = write_dir + "/val.csv"

        train_data_list[i].to_csv(write_path_train, index=False)
        val_data_list[i].to_csv(write_path_val, index=False)

    data.to_csv("./data_path/train_full.csv", index=False)

