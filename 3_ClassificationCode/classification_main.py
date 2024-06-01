
import sys
sys.path.append('./lib')
from utils import *
from my_dataset import CustomTableDataset
import pandas as pd
import numpy as np
from lib.cal_metrics import cal_metrics_from_table

def classification(feattype):
    K = 10
    acc_list = []
    for i in range(K):
        train_df = pd.read_csv("../2_FeatureSelectionCode/data/path" + str(i) + "/concat/train/data_" + feattype + ".csv")
        val_df = pd.read_csv("../2_FeatureSelectionCode/data/path" + str(i) + "/concat/val/data_" + feattype + ".csv")
        val_acc = full_procedure(train_df, val_df, i, feattype)
        acc_list.append(val_acc)

    print(acc_list)
    avg = sum(acc_list) / len(acc_list)

    print("acc_score:{}%".format(avg))

def full_proc_table(dir_path, k=10):
    df = pd.DataFrame(columns=['acc', 'recall', 'precition', 'f1', 'mcc'])
    for i in range(k):
        read_path = dir_path + "/label_" + str(i) +".csv"
        data_metrics = pd.read_csv(read_path)
        results_metrics = cal_metrics_from_table(data_metrics)
        df.loc[i, ['acc', 'recall', 'precition', 'f1', 'mcc']] = results_metrics

    return list(df.mean())


if __name__ == "__main__":
    classification("full")

    results_full_dir = "./results/full"

    data_results = pd.DataFrame(columns=["feat_name", 'acc', 'recall', 'precition', 'f1', 'mcc'])

    results_metrics = full_proc_table(dir_path=results_full_dir, k=10)
    results = ["full"] + results_metrics
    data_results.loc[0, ["feat_name", 'acc', 'recall', 'precition', 'f1', 'mcc']] = results

    data_results.to_csv("./results_full.csv", index=False)