# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
import torch
from PIL import Image

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_my_datasets(is_train, df, args):
    transform = build_transform(is_train, args)

    dataset = CustomDataset(df, transform=transform)

    print(dataset)

    return dataset


class CustomDataset(Dataset):

    def __init__(self, df, transform=None):
        """
        df：Pandas DataFrame，with columns named "label"
        trasnform：data preprocessing
        """
        super().__init__()
        self.path_label = df
        self.transform = transform

    def __len__(self):
        # get the number of samples
        return self.path_label.shape[0]

    def __info__(self):
        print("bio_img data")
        print("\t Number of Samples: {}".format(self.path_label.shape[0]))
        print("\t Number of patients: {}".format(len(self.path_label["patient_id"].unique())))

    def __getitem__(self, idx):
        # make sure idx is not a tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.path_label["path"].values[idx]
        image = Image.open(image_path)
        # image = io.imread(image_path)

        label = self.path_label["label"].values[idx]

        if self.transform:
            # image = Image.fromarray(io.imread(image_path))
            image = self.transform(image)

        return image, label


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
