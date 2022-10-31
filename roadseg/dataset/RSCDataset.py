import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .transform import train_transform
from .transform import val_transform


#
#
# class RSCDataset(Dataset):
#     def __init__(self, imgs_dir, masks_dir, transform=None):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.transform = transform
#         self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
#                     if not file.startswith('.')]
#         logging.info(f'Creating dataset with {len(self.ids)} examples')
#
#     def __len__(self):
#         return len(self.ids)
#
#     @classmethod
#     def preprocess(cls, pil_img):
#         img_nd = np.array(pil_img)
#         if len(img_nd.shape) == 2:
#             img_nd = np.expand_dims(img_nd, axis=2)
#         try:
#             img_trans = img_nd.transpose(2, 0, 1)
#         except:
#             print(img_nd.shape)
#         if img_trans.max() > 1: img_trans = img_trans / 255
#         return img_trans
#
#     def __getitem__(self, i):
#         idx = self.ids[i]
#         img_file = glob(self.imgs_dir + idx + '.*')
#         mask_file = glob(self.masks_dir + idx + '.*')
#
#         image = cv2.imread(img_file[0], cv2.IMREAD_COLOR)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)
#
#         # if self.transform:
#         transformed = self.transform(image=image, mask=mask)
#         image = transformed['image'] / 1
#         mask = transformed['mask'] / 255
#
#         return {
#             'image': image,
#             'label': mask
#         }


def merge_picture(src_path, filename, num_yx=4):
    i = 0
    list_a = []
    for name in filename:
        # i用于计数
        i += 1
        # t用于换列
        t = (i - 1) // num_yx
        # 获取img
        img = Image.open(os.path.join(src_path, name))
        if img.size[0] != 512:
            img = img.resize((512, 512), Image.ANTIALIAS)
            # 转换为numpy数组
        im_array = np.array(img)
        if (i - 1) % num_yx == 0:
            # list_a[t] = im_array
            list_a.append(im_array)
        else:
            list_a[t] = np.concatenate((list_a[t], im_array), axis=0)
    # 2 合成列以后需要将列都拼接起来
    for j in range(len(list_a) - 1):
        list_a[0] = np.concatenate((list_a[0], list_a[j + 1]), axis=1)
    return np.uint8(list_a[0])


class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self):
        return 4000

    def __getitem__(self, index):
        filenames = os.listdir(self.imgs_dir)
        sample = random.sample(filenames, 16)  # 随机选取picknumber数量的样本图片
        print(sample)
        image = merge_picture(self.imgs_dir, sample)
        mask = merge_picture(self.masks_dir, sample)
        # if self.transform:
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image'] / 1
        mask = transformed['mask'] / 255
        return {
            'image': image,
            'label': mask
        }


if __name__ == '__main__':
    data_dir = "/media/limzero/qyl/mmsegmentation/data/satellite_jpg/"
    train_imgs_dir = os.path.join(data_dir, "img_dir/train/")
    train_labels_dir = os.path.join(data_dir, "ann_dir/train/")

    train_data = RSCDataset(train_imgs_dir, train_labels_dir, transform=train_transform)
    train_data[500]
