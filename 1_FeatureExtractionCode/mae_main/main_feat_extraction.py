
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import PIL

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import models_vit
from models_vit import vit_base_patch16

from engine_pretrain import train_one_epoch
from util.datasets import CustomDataset


def Extract_single_image(img_path, model, transform, device):
    image = Image.open(img_path)

    image = transform(image)

    # expand batch dimension
    image = torch.unsqueeze(image, dim=0)

    with torch.no_grad():
        feat = model.forward_features(image.to(device))

    return feat.cpu().numpy().flatten()

def Extract_whole_set(df, weight_path, featlen=768):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = vit_base_patch16()

    # print(model)
    model = model.to(device)
    ckpt = torch.load(weight_path, map_location=device)["model"]
    ckpt.pop('head.weight')
    ckpt.pop('head.bias')

    model.load_state_dict(ckpt, strict=False)

    model.eval()

    t = []
    input_size = 224
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    data_transform = transforms.Compose(t)

    column_name_list = []
    for i in range(1, featlen + 1):
        column_name = "feat" + str(i)
        column_name_list.append(column_name)

    feat_df = pd.DataFrame(columns=column_name_list)

    for idx in range(df.shape[0]):
        img_path = df.loc[idx, ["path"]].values
        feat = Extract_single_image(img_path=img_path[0], model=model, transform=data_transform, device=device)
        feat_df.loc[idx, column_name_list] = feat

    df = pd.concat([df, feat_df], axis=1)

    return df

def feat_extract(dir_path):
    train_csv_path = dir_path + "/train.csv"
    val_csv_path = dir_path + "/val.csv"
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    weight_path = dir_path + "/finetune_outputdir/checkpoint-249.pth"

    feat_train_df = Extract_whole_set(df=train_df, weight_path=weight_path)
    feat_val_df = Extract_whole_set(df=val_df, weight_path=weight_path)

    write_dir = dir_path + "/data"
    if os.path.exists(write_dir) is False:
        os.makedirs(write_dir)

    train_write_path = write_dir + "/train.csv"
    val_write_path = write_dir + "/val.csv"

    feat_train_df.to_csv(train_write_path, index=False)
    feat_val_df.to_csv(val_write_path, index=False)


if __name__ == '__main__':
    K = 10
    for i in range(K):
        print(i)
        print("encoding the %d sets" %i)
        dir_path = "./data_path/path" + str(i)
        feat_extract(dir_path)

