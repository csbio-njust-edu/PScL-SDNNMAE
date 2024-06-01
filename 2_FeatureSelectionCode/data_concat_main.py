
import pandas as pd
import numpy as np
from scipy.io import loadmat
import os
from scipy.io import savemat

def load_mat(mat_file_path):
    data = loadmat(mat_file_path)
    featkey = list(data.keys())[3]
    data = data[featkey]
    data[np.isnan(data) | np.isinf(data)] = 0
    data = pd.DataFrame(data)
    return data

def rebuilt_column_name(df, feat_name):
    column_name_list = []
    for i in range(1, df.shape[1]+1):
        column_name = feat_name + str(i)
        column_name_list.append(column_name)
    df.columns = column_name_list
    return df

def concat_features(read_dir_path, is_train=True):
    if is_train:
        clbp_sda_path = read_dir_path + "/sda/train/CLBP_sda.mat"
        lbp_sda_path = read_dir_path + "/sda/train/LBP_sda.mat"
        let_sda_path = read_dir_path + "/sda/train/LET_sda.mat"
        riclbp_sda_path = read_dir_path + "/sda/train/RICLBP_sda.mat"
        slfs_sda_path = read_dir_path + "/sda/train/SLFs_sda.mat"
        mae_sda_path = read_dir_path + "/pure/train/mae.csv"
        label_path = read_dir_path + "/am/train/label.mat"
    else:
        clbp_sda_path = read_dir_path + "/sda/val/CLBP_sda.mat"
        lbp_sda_path = read_dir_path + "/sda/val/LBP_sda.mat"
        let_sda_path = read_dir_path + "/sda/val/LET_sda.mat"
        riclbp_sda_path = read_dir_path + "/sda/val/RICLBP_sda.mat"
        slfs_sda_path = read_dir_path + "/sda/val/SLFs_sda.mat"
        mae_sda_path = read_dir_path + "/pure/val/mae.csv"
        label_path = read_dir_path + "/am/val/label.mat"

    data_clbp = load_mat(clbp_sda_path)
    data_lbp = load_mat(lbp_sda_path)
    data_riclbp = load_mat(riclbp_sda_path)
    data_let = load_mat(let_sda_path)
    data_slfs = load_mat(slfs_sda_path)
    data_mae = pd.read_csv(mae_sda_path)
    data_label = load_mat(label_path)
    data_label = data_label.T
    data_label.columns = ['label']
    data_label = data_label - 1

    data_full = pd.concat([data_clbp, data_lbp, data_riclbp, data_let, data_slfs, data_mae], axis=1)
    data_no_mae = pd.concat([data_clbp, data_lbp, data_riclbp, data_let, data_slfs], axis=1)
    if is_train:
        write_dir_path = read_dir_path + "/concat/train"
    else:
        write_dir_path = read_dir_path + "/concat/val"

    if os.path.exists(write_dir_path) is False:
        os.makedirs(write_dir_path)

    data_full = rebuilt_column_name(data_full, "feat")
    data_full = pd.concat([data_full, data_label], axis=1)
    data_full.to_csv(write_dir_path + "/data_full.csv", index=False)
    print(data_full.shape)

    data_no_mae = rebuilt_column_name(data_no_mae, "feat")
    data_no_mae = pd.concat([data_no_mae, data_label], axis=1)
    data_no_mae.to_csv(write_dir_path + "/data_no_mae.csv", index=False)
    print(data_no_mae.shape)

if __name__ == "__main__":
    K = 10
    for i in range(K):
        print(i)
        read_dir_path = "./data/path" + str(i)
        concat_features(read_dir_path=read_dir_path, is_train=True)  # concatenate the train data
        concat_features(read_dir_path=read_dir_path, is_train=False)  # concatenate the validation data
