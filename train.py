#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： LUO Elio
# datetime： 2023/9/10 23:07
# ide： PyCharm

import os
from trainer import trainn
import glob
from nets import trans_deeplab
import math
import torch


if __name__ == "__main__":

    net = trans_deeplab.DeepLab(num_classes=2, backbone="mobilenet", downsample_factor=32, pretrained=True).cuda()

    num_classes = 2
    batch_size = 4
    image_paths = glob.glob('datadir\*tif')
    label_paths =glob.glob('labeldir\*tif')

    epoch = 100

    lr = 2e-5

    optimizer = torch.optim.AdamW(net.parameters(),
                                  lr=lr,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.01,
                                  )

    def split_train_val(image_paths, label_paths, val_index=0):
        train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
        for i in range(len(image_paths)):
            if i % 10 == val_index:
                val_image_paths.append(image_paths[i])
                val_label_paths.append(label_paths[i])
            else:
                train_image_paths.append(image_paths[i])
                train_label_paths.append(label_paths[i])
        print("Number of train images: ", len(train_image_paths))
        print("Number of val images: ", len(val_image_paths))
        return train_image_paths, train_label_paths, val_image_paths, val_label_paths

    image_path = 'datadir'

    train_num = len(os.listdir(image_path))
    val_num = math.ceil(len(os.listdir(image_path)) // 10)
    epoch_step = train_num // batch_size
    epoch_step_val = val_num // batch_size
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val(image_paths,
                                                                                             label_paths,
                                                                                             0)

    trainn(num_classes, net, train_image_paths, val_image_paths, epoch_step, epoch_step_val, train_label_paths,
           val_label_paths, epoch, optimizer, batch_size)
