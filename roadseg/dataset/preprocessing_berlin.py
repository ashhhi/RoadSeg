# berlin dataset: https://zenodo.org/record/1154821#.ZDGauuZBztX

import os
import shutil
import cv2 as cv
import numpy as np
from progressbar import ProgressBar

path = r'C:\DataSet\berlin'

#
def mymovefile(srcfile,dstpath):                       # 移动函数
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath, fname = os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, os.path.join(dstpath, fname))          # 移动文件
        print ("move %s -> %s"%(srcfile, os.path.join(dstpath, fname)))


print("========================分类存放gt和train========================")
train_path = os.path.join(path, 'train')
gt_path = os.path.join(path, 'gt')

files = os.listdir(path)

if len(files) != 2:
    for file in files:
        file_name = file.split('_')[1].split('.')[0]
        if file_name == 'image':
            mymovefile(os.path.join(path, file), train_path)

        else:
            mymovefile(os.path.join(path, file), gt_path)

print("========================改变label颜色========================")
gt_file = os.listdir(gt_path)

pbar = ProgressBar()
for iterate in pbar(range(len(gt_file))):
    file = gt_file[iterate]
    img_gray = cv.imread(os.path.join(gt_path, file), cv.IMREAD_GRAYSCALE)
    ret, img_binary = np.array(cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY_INV))
    cv.imwrite(os.path.join(gt_path, file), img_binary)


print("========================resize和rename========================")

# gt和train都resize和rename
def list_dir(file_dir):
    '''
        通过 listdir 得到的是仅当前路径下的文件名，不包括子目录中的文件，如果需要得到所有文件需要递归
    '''
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        # 获取文件的绝对路径
        path = os.path.join(file_dir, cur_file)
        new_name = cur_file.split('_')[0]+'.png'
        if os.path.isfile(path):
            img = cv.imread(path)
            img = cv.resize(img, (1456, 1140))
            cv.imwrite(path, img)
            os.rename(path, os.path.join(file_dir, new_name))
        else:
            # print("{0} : is dir!".format(cur_file))
            list_dir(path) # 递归子目录


list_dir(path)

