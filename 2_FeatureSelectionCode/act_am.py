
import os

from lib.AM import standardize_dataframe, Anova_M_Info

import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat

def load_mat(mat_file_path):
    data = loadmat(mat_file_path)
    featkey = list(data.keys())[3]
    data = data[featkey]
    data[np.isnan(data) | np.isinf(data)] = 0
    data = pd.DataFrame(data)
    # data = data.flatten()
    return data

def load_mat_array(mat_file_path):
    data = loadmat(mat_file_path)
    featkey = list(data.keys())[3]
    data = data[featkey]
    data[np.isnan(data) | np.isinf(data)] = 0
    # data = pd.DataFrame(data)
    data = data.flatten()
    return data

def rebuilt_column_name(df, feat_name):
    column_name_list = []
    for i in range(1, df.shape[1]+1):
        column_name = feat_name + str(i)
        column_name_list.append(column_name)
    df.columns = column_name_list
    return df

def build_path(path0, feattype):
    start_ = path0[:3]
    start_ = start_ + "1_FeatureExtractionCode/data/3_features/"
    mid = feattype + "/lin_db1/"
    main_ = path0[16:-3]
    tail = "mat"
    final_path = start_ + mid + main_ + tail
    return final_path

def built_feature_set(path_df, feattype, featlen):
    column_name_list = []
    for i in range(1, featlen + 1):
        column_name = feattype + str(i)
        column_name_list.append(column_name)

    df = pd.DataFrame(columns=column_name_list)

    for idx in range(path_df.shape[0]):
        mat_path = build_path(path0=path_df.loc[idx, "path"], feattype=feattype)
        x = load_mat_array(mat_path)
        df.loc[idx, column_name_list] = x

    return df

def save_am_sets(read_dir_path, write_dir_path):
    write_dir_path1 = write_dir_path
    read_csv_path = read_dir_path + "/train.csv"
    mae_path = read_dir_path + "/data/train.csv"
    write_dir_path = write_dir_path + "/am/train"

    path_df = pd.read_csv(read_csv_path)

    data_slfs_lbp = built_feature_set(path_df=path_df, feattype="SLFs_LBPs", featlen=1097)

    data_slfs = data_slfs_lbp.iloc[:, :841]
    data_lbp = data_slfs_lbp.iloc[:, 841:]
    data_clbp = built_feature_set(path_df=path_df, feattype="CLBP", featlen=906)
    data_riclbp = built_feature_set(path_df=path_df, feattype="RICLBP", featlen=408)
    data_let = built_feature_set(path_df=path_df, feattype="LET", featlen=413)

    data_slfs = rebuilt_column_name(data_slfs, "SLFs")
    data_lbp = rebuilt_column_name(data_lbp, "LBP")

    # mae
    data_mae = pd.read_csv(mae_path)
    data_mae = data_mae.iloc[:, 2:]

    if os.path.exists(write_dir_path) is False:
        os.makedirs(write_dir_path)

    label = np.array(path_df['label'])
    print(label.shape)

    selected_cols_slfs = Anova_M_Info(data_slfs, label)
    selected_cols_lbp = Anova_M_Info(data_lbp, label)
    selected_cols_riclbp = Anova_M_Info(data_riclbp, label)
    selected_cols_let = Anova_M_Info(data_let, label)
    selected_cols_clbp = Anova_M_Info(data_clbp, label)
    selected_cols_mae = Anova_M_Info(data_mae, label)

    # for matlab form
    label = np.array(path_df['label'] + 1)

    df_slfs_am = data_slfs.loc[:, selected_cols_slfs]
    df_lbp_am = data_lbp.loc[:, selected_cols_lbp]
    df_clbp_am = data_clbp.loc[:, selected_cols_clbp]
    df_riclbp_am = data_riclbp.loc[:, selected_cols_riclbp]
    df_let_am = data_let.loc[:, selected_cols_let]
    df_mae_am = data_mae.loc[:, selected_cols_mae]

    # print(df_slfs_am.shape)
    # print(len(selected_cols_slfs))

    data_slfs_save = {
        'SLFs': np.array(df_slfs_am).astype(np.float64)
    }
    savemat(write_dir_path + '/SLFs_am.mat', data_slfs_save)

    data_lbp_save = {
        "LBP": np.array(df_lbp_am).astype(np.float64)
    }
    savemat(write_dir_path + '/LBP_am.mat', data_lbp_save)

    data_clbp_save = {
        "CLBP": np.array(df_clbp_am).astype(np.float64)
    }
    savemat(write_dir_path + '/CLbp_am.mat', data_clbp_save)

    data_riclbp_save = {
        "RICLBP": np.array(df_riclbp_am).astype(np.float64)
    }
    savemat(write_dir_path + '/RICLBP_am.mat', data_riclbp_save)

    data_let_save = {
        "LET": np.array(df_let_am).astype(np.float64)
    }
    savemat(write_dir_path + '/LET_am.mat', data_let_save)

    data_mae_save = {
        "MAE": np.array(df_mae_am).astype(np.float64)
    }
    savemat(write_dir_path + '/MAE_am.mat', data_mae_save)

    data_label_save = {
        "label": label.astype(np.float64)
    }
    savemat(write_dir_path + '/label.mat', data_label_save)


    ##### val set

    read_csv_path = read_dir_path + "/val.csv"
    mae_path = read_dir_path + "/data/val.csv"
    write_dir_path = write_dir_path1 + "/am/val"

    path_df = pd.read_csv(read_csv_path)

    data_slfs_lbp = built_feature_set(path_df=path_df, feattype="SLFs_LBPs", featlen=1097)

    data_slfs = data_slfs_lbp.iloc[:, :841]
    data_lbp = data_slfs_lbp.iloc[:, 841:]
    data_clbp = built_feature_set(path_df=path_df, feattype="CLBP", featlen=906)
    data_riclbp = built_feature_set(path_df=path_df, feattype="RICLBP", featlen=408)
    data_let = built_feature_set(path_df=path_df, feattype="LET", featlen=413)

    data_slfs = rebuilt_column_name(data_slfs, "SLFs")
    data_lbp = rebuilt_column_name(data_lbp, "LBP")

    # mae
    data_mae = pd.read_csv(mae_path)
    data_mae = data_mae.iloc[:, 2:]

    if os.path.exists(write_dir_path) is False:
        os.makedirs(write_dir_path)

    label = np.array(path_df['label'] + 1)

    df_slfs_am = data_slfs.loc[:, selected_cols_slfs]
    df_lbp_am = data_lbp.loc[:, selected_cols_lbp]
    df_clbp_am = data_clbp.loc[:, selected_cols_clbp]
    df_riclbp_am = data_riclbp.loc[:, selected_cols_riclbp]
    df_let_am = data_let.loc[:, selected_cols_let]
    df_mae_am = data_mae.loc[:, selected_cols_mae]

    data_slfs_save = {
        'SLFs': np.array(df_slfs_am).astype(np.float64)
    }
    savemat(write_dir_path + '/SLFs_am.mat', data_slfs_save)

    data_lbp_save = {
        "LBP": np.array(df_lbp_am).astype(np.float64)
    }
    savemat(write_dir_path + '/LBP_am.mat', data_lbp_save)

    data_clbp_save = {
        "CLBP": np.array(df_clbp_am).astype(np.float64)
    }
    savemat(write_dir_path + '/CLbp_am.mat', data_clbp_save)

    data_riclbp_save = {
        "RICLBP": np.array(df_riclbp_am).astype(np.float64)
    }
    savemat(write_dir_path + '/RICLBP_am.mat', data_riclbp_save)

    data_let_save = {
        "LET": np.array(df_let_am).astype(np.float64)
    }
    savemat(write_dir_path + '/LET_am.mat', data_let_save)

    data_mae_save = {
        "MAE": np.array(df_mae_am).astype(np.float64)
    }
    savemat(write_dir_path + '/MAE_am.mat', data_mae_save)

    data_label_save = {
        "label": label.astype(np.float64)
    }
    savemat(write_dir_path + '/label.mat', data_label_save)


if __name__ == '__main__':
    if os.path.exists("./data") is False:
        os.makedirs("./data")

    K = 10
    for i in range(K):
        print(i)
        read_dir_path = "../1_FeatureExtractionCode/mae_main/data_path/path" + str(i)
        write_dir_path = "./data/path" + str(i)
        save_am_sets(read_dir_path=read_dir_path, write_dir_path=write_dir_path)