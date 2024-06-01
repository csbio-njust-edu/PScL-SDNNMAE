import torchvision
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import models as m
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import pandas as pd
import datetime
import gc
from my_dataset import CustomTableDataset
from model import DNN_4l


class EarlyStopping():
    def __init__(self, patience=15, tol=0.0000005):
        # if the loss does not decrease continuously for "patience" times
        # activate early stop

        self.patience = patience
        self.tol = tol  # tolerance
        self.counter = 0  # count
        self.lowest_loss = None
        self.early_stop = False  # True - early stop

    def __call__(self, val_loss):
        if self.lowest_loss == None:
            self.lowest_loss = val_loss
        elif self.lowest_loss - val_loss > self.tol:
            self.lowest_loss = val_loss
            self.counter = 0
        elif self.lowest_loss - val_loss < self.tol:
            self.counter += 1
            # print("\t NOTICE: Early stopping counter {} of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                # print('\t NOTICE: Early Stopping Actived')
                self.early_stop = True
        return self.early_stop


def IterOnce(net, criterion, opt, x, y):
    """
    for one iteration for the model

    net: architecture
    criterion: loss function
    opt: optimizer
    x: all samples of a batch
    y: all labels of a batch
    """
    sigma = net.forward(x)
    loss = criterion(sigma, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)  # 比起设置梯度为0，让梯度为None会更节约内存
    yhat = torch.max(sigma, 1)[1]
    correct = torch.sum(yhat == y)
    return correct, loss


def TestOnce(net, criterion, x, y):
    """
    test on one iteration for the model

    net: architecture
    criterion：loss function
    x：all samples of a batch
    y：all labels of a batch
    """
    with torch.no_grad():
        sigma = net.forward(x)
        loss = criterion(sigma, y)
        yhat = torch.max(sigma, 1)[1]
        correct = torch.sum(yhat == y)
    return correct, loss


def fit_test(device, net, batchdata, testdata, criterion, opt, scheduler, epochs, tol):
    # for training a model
    Sample_Per_Epoch = batchdata.dataset.__len__()
    allsamples = Sample_Per_Epoch * epochs
    trainedsamples = 0

    trainlosslist = []
    testlosslist = []
    trainacclist = []
    testacclist = []
    early_stopping = EarlyStopping(tol=tol)
    highestacc = None

    if os.path.exists("./tmp") is False:
        os.makedirs("./tmp")

    for epoch in range(1, epochs + 1):
        net.train()
        loss_train = 0
        correct_train = 0
        for batch_idx, (x, y) in enumerate(batchdata):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(x.shape[0])
            correct, loss = IterOnce(net, criterion, opt, x, y)
            trainedsamples += x.shape[0]
            correct_train += correct
            loss_train += loss

            if (batch_idx + 1) % 125 == 0:
                print('Epoch{}:[{}/{}({:.0f}%)]'.format(epoch, trainedsamples, allsamples,
                                                        100 * trainedsamples / allsamples))
        scheduler.step()

        TrainAcc = float(correct_train * 100) / Sample_Per_Epoch
        TrainLoss = float(loss_train) / Sample_Per_Epoch
        trainlosslist.append(TrainLoss)
        trainacclist.append(TrainAcc)

        del x, y, correct, loss, correct_train, loss_train
        gc.collect()
        torch.cuda.empty_cache()

        net.eval()
        loss_test = 0
        correct_test = 0
        test_sample = testdata.dataset.__len__()

        for x, y in testdata:
            with torch.no_grad():
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).view(x.shape[0])
                correct, loss = TestOnce(net, criterion, x, y)
                loss_test += loss
                correct_test += correct

        TestAcc = float(correct_test * 100) / test_sample
        TestLoss = float(loss_test) / test_sample
        testlosslist.append(TestLoss)
        testacclist.append(TestAcc)

        del x, y, correct, loss, correct_test, loss_test
        gc.collect()
        torch.cuda.empty_cache()

        print("\t Train Loss:{:.6f}, Test Loss:{:.6f}, Train Acc:{:.3f}%, Test Acc:{:.3f}%".format(TrainLoss,
                                                                                                   TestLoss,
                                                                                                   TrainAcc,
                                                                                                   TestAcc))

        write_path = "./tmp/dnn_4l.pth"
        if highestacc == None:
            highestacc = TestAcc
        if highestacc < TestAcc:
            highestacc = TestAcc
            torch.save(net.state_dict(), write_path)
            # torch.save(opt.state_dict(), os.path.join(PATH, opt_name + '.pt'))
            # print('\t weight Saved')

        early_stop = early_stopping(TestLoss)
        if early_stop == True:
            break
        # break # 1 epoch

    # print('Done')
    # print(time.time() - start_time)
    # torch.save(net.state_dict(), os.path.join(PATH, model_epoch + '.pth'))
    # torch.save(opt.state_dict(), os.path.join(PATH, opt_epoch + '.pt'))
    return trainlosslist, testlosslist, trainacclist, testacclist,

def predict(weight_path, val_df, device, transform, write_path):
    """
    predict the validation set using trained model

    weight_path: the path of trained weight
    val_df: the validation set data
    device: device to use
    transform: the transformation
    write_path: write path of the results
    """
    model = DNN_4l(input_dim=val_df.shape[1] - 1, num_classes=7).to(device, non_blocking=True)
    model = model.to(device, non_blocking=True)
    model.load_state_dict(torch.load(weight_path, map_location=device))

    x = val_df.iloc[:, :-1]

    model.eval()
    features = torch.tensor(np.array(x), dtype=torch.float32)
    features = transform(features)
    features = features.to(device, non_blocking=True)

    with torch.no_grad():
        output = torch.squeeze(model(features)).cpu()
        predict = torch.softmax(output, dim=0)
        yhat = torch.max(predict, 1)[1].numpy()

    data_df = pd.DataFrame(columns=['y', 'yhat'])
    data_df['y'] = val_df['label']
    data_df['yhat'] = np.array(yhat)

    column_name_list = []
    for i in range(1, 8):
        column_name = "prob" + str(i)
        column_name_list.append(column_name)
    df3 = pd.DataFrame(np.array(predict), columns=column_name_list)
    data_df = data_df.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)
    data_df2 = pd.concat([data_df, df3], axis=1)

    data_df2.to_csv(write_path, index=False)

def save_iter_result(trainlosslist, testlosslist, trainacclist, testacclist, save_path="./iter_result.csv"):

    df = pd.DataFrame(columns=['trainloss', 'testloss', 'trainacc', 'testacc'])
    df['trainloss'] = trainlosslist
    df['testloss'] = testlosslist
    df['trainacc'] = trainacclist
    df['testacc'] = testacclist

    df.to_csv(save_path, index=False)


def calculate_mean_std(df):
    """
    calculate the mean and std for all the features

    df: dataframe
    mean: the vector pf mean
    std: the vector of std
    """
    mean = df.mean()
    std = df.std()

    return mean, std


def standardize_tensor(features, mean, std):
    standardized_features = (features - mean) / std
    return standardized_features


def full_procedure(train_df, test_df, idx, feattype, epochs=50, bs=64, lr=0.0001, betas=(0.9, 0.99), wd=0, tol=10 ** (-5)):
    torch.manual_seed(1412)
    torch.cuda.manual_seed(1412)
    torch.cuda.manual_seed_all(1412)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = train_df.astype("float32")
    test_df = test_df.astype("float32")

    mean, std = calculate_mean_std(train_df.iloc[:, :-1])

    data_transform = {
        "train": transforms.Compose([
            transforms.Lambda(lambda x: standardize_tensor(x, mean=torch.tensor(mean), std=torch.tensor(std)))
            # transforms.Lambda(lambda x: add_noise_tensor(x, noise_level=0.1))
        ]),
        "val": transforms.Compose([
            transforms.Lambda(lambda x: standardize_tensor(x, mean=torch.tensor(mean), std=torch.tensor(std)))
            # transforms.Lambda(lambda x: add_noise_tensor(x, noise_level=0.1))
        ])}

    train_dataset = CustomTableDataset(train_df, transform=data_transform["train"])
    val_dataset = CustomTableDataset(test_df, transform=data_transform["val"])

    lr_lambda = lambda epoch: 1.0 ** epoch

    nw = min([os.cpu_count(), bs if bs > 1 else 0, 16])

    batchdata = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=False, pin_memory=True, num_workers=nw)
    testdata = DataLoader(val_dataset, batch_size=bs, shuffle=False, drop_last=False, pin_memory=True, num_workers=nw)

    model = DNN_4l(input_dim=train_df.shape[1] - 1, num_classes=7).to(device, non_blocking=True)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion = criterion.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=wd, amsgrad=True)
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda)

    dir_path = "./results/" + feattype

    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)

    trainloss, testloss, trainacc, testacc = fit_test(device=device
                                                      , net=model
                                                      , batchdata=batchdata
                                                      , testdata=testdata
                                                      , criterion=criterion
                                                      , opt=opt
                                                      , scheduler=scheduler
                                                      , epochs=epochs
                                                      , tol=tol
                                                      )
    write_csv_path = dir_path + "/label_" + str(idx) + ".csv"
    predict(weight_path="./tmp/dnn_4l.pth", val_df=test_df, device=device,
            transform=data_transform['val'], write_path=write_csv_path)

    acc_score = max(testacc)
    print("accuracy:", acc_score)
    return acc_score






