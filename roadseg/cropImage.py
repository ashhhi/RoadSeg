# coding: utf-8
from PIL import Image
import os
import os.path
import numpy as np
import cv2

# 指明被遍历的文件夹
rootdir = r'D:\SAR/SAR/trainData/image'
for parent, dirnames, filenames in os.walk(rootdir):  # 遍历每一张图片
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print(currentPath)
        gtPath = currentPath.replace('image', 'gt')
        print('the full name of the file is :' + currentPath)
        prefix = filename.split('.')[0]
        img = Image.open(currentPath)
        gt = Image.open(gtPath)
        print(img.format, img.size, img.mode)

        if np.mean(gt) < 0:
            img.save("dataset/train/image/" + filename)  # 存储裁剪得到的图像
            gt.save("dataset/train/gt/" + filename)  # 存储裁剪得到的图像
        # img.show()
        else:
            box1 = (0, 0, img.size[1] // 2, img.size[0] // 2)  # 设置左、上、右、下的像素
            image1 = img.crop(box1)  # 图像裁剪
            image1.save("dataset/train/image/" + prefix + '-1.png')  # 存储裁剪得到的图像
            gt1 = gt.crop(box1)  # 图像裁剪
            gt1.save("dataset/train/gt/" + prefix + '-1.png')  # 存储裁剪得到的图像

            box2 = (0, img.size[0] // 2, img.size[1] // 2, img.size[0])  # 设置左、上、右、下的像素
            image2 = img.crop(box2)  # 图像裁剪
            image2.save("dataset/train/image/" + prefix + '-2.png')  # 存储裁剪得到的图像
            gt2 = gt.crop(box2)  # 图像裁剪
            gt2.save("dataset/train/gt/" + prefix + '-2.png')  # 存储裁剪得到的图像

            box3 = (img.size[1] // 2, 0, img.size[1], img.size[0] // 2)  # 设置左、上、右、下的像素
            image3 = img.crop(box3)  # 图像裁剪
            image3.save("dataset/train/image/" + prefix + '-3.png')  # 存储裁剪得到的图像
            gt3 = gt.crop(box3)  # 图像裁剪
            gt3.save("dataset/train/gt/" + prefix + '-3.png')  # 存储裁剪得到的图像

            box4 = (img.size[0] // 2, img.size[1] // 2, img.size[0], img.size[1])  # 设置左、上、右、下的像素
            image4 = img.crop(box4)  # 图像裁剪
            image4.save("dataset/train/image/" + prefix + '-4.png')  # 存储裁剪得到的图像
            gt4 = gt.crop(box4)  # 图像裁剪
            gt4.save("dataset/train/gt/" + prefix + '-4.png')  # 存储裁剪得到的图像