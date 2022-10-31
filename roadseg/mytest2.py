# -*- coding:utf-8 -*-
# import os
#
# import numpy as np
# import skimage.transform as trans
# from skimage import img_as_ubyte


# from PIL import Image
# from skimage import io

#
# ############测试阶段#############
#
#
#
#
#
#
# # 进行测试
# def NetModel(imgpath, savepath, modelname,number):
#     testGene = testGenerator(imgpath, number, False, False,(256,256))
#     model = load_model(modelname)#,custom_objects={'MaxPoolingWithArgmax2D': MaxPoolingWithArgmax2D, 'MaxUnpooling2D': MaxUnpooling2D}
#     results = model.predict_generator(testGene, number, verbose=1)
#     saveResult(savepath, results)
#     return 0
#
#
# # 保存测试结果
# def saveResult(save_path, npyfile, flag_multi_class=False, num_class=1):
#     for i, item in enumerate(npyfile):
#         mask_pred = item[:, :, 0]
#         mask_pred = np.round(mask_pred)  # 返回浮点数的四舍五入值
#         io.imsave(os.path.join(save_path, "%d_predict.tif" % (i + 1)), img_as_ubyte(mask_pred))
#
#

#
#
# NetModel(imgpath=r"E:\shanfangmei\息肉分割\息肉数据集\test\image", savepath=r"E:\shanfangmei\息肉分割\unet2\predict", modelname=r'E:\shanfangmei\息肉分割\unet2\unet.hdf5',number=153)
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
from PIL import Image
import matplotlib.image as mpimg
import skimage.io
import torchvision.transforms as transforms
from datetime import datetime


# 评价指标
def sorensen_dices(y_true, y_pred):  # 计算三个评价指标
    y_pred = np.round(y_pred)
    y_true = np.round(y_true)
    intersection = np.sum(y_true * y_pred)
    TN = np.sum((1 - y_true) * (1 - y_pred))
    DSI = (2 * intersection + 1) / (np.sum(y_true) + np.sum(y_pred) + 1)

    Re_y_true = np.ones(y_true.shape)
    Re_y_true[y_true > 0] = 0
    Re_y_pred = np.ones(y_pred.shape)
    Re_y_pred[y_pred > 0] = 0
    ReS_intse = np.sum(Re_y_true * y_pred)
    ReT_intse = np.sum(y_true * Re_y_pred)
    if (np.sum(y_true) + np.sum(y_pred)) == 0:
        FPD = 0
        FND = 0
        print('hh')
    else:
        FPD = 2 * ReS_intse / (np.sum(y_true) + np.sum(y_pred))
        FND = 2 * ReT_intse / (np.sum(y_true) + np.sum(y_pred))

    # Jaccard相似度
    x = y_pred + y_true
    union = np.sum(x > 0)
    Js = (intersection + 1) / (union + 1)

    #精确度precision
    precision=(intersection+1)/(np.sum(y_pred)+1)

    # 特异性
    specificity = (TN+1) / (np.sum(1 - y_true)+1)

    #灵敏度sensitivity
    sensitivity=(intersection+1)/(np.sum(y_true)+1)

    #正确率
    accuary=(TN+intersection+1) / (np.sum(1 - y_true)+np.sum(y_true)+1)

    #f1分数
    f1=(2*precision*sensitivity+1)/(precision+sensitivity+1)

    iou=(intersection+1)/(np.sum(y_true)+1)

    return DSI, FPD, FND, Js, precision, specificity, sensitivity,accuary,f1,iou


# 进行评估，获得评估结果
def Evaluation(Predict_Path, Label_Path, target_size, Begin=0, End=0):  # 输出三个评测指标（整体上）
    DSIs = 0
    FPDs = 0
    FNDs = 0
    Jss = 0
    precisions=0
    specificitys=0
    sensitivitys=0
    accuarys=0
    f1s = 0
    ious = 0
    sum = End - Begin + 1
    transform1 = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    )

    for k in range(End):
        y_true = Image.open(os.path.join(Label_Path, "%d.png" % k))#0~255 array
        y_true = y_true.convert('L')
        y_true = transform1(y_true)
        zero = torch.zeros_like(y_true)
        one = torch.ones_like(y_true)
        y_true = torch.where(y_true > 0.5, one, zero)
        y_true = y_true.numpy().squeeze()
        y_true = y_true.astype('float32')#(256,256)  0、1
        #y_pred = skimage.io.imread(os.path.join(Predict_Path, "%d.png" % k))#0、255 array  (256,256)

        y_pred = Image.open(os.path.join(Predict_Path, "%d.png" % k))
        y_pred = y_pred.convert('L')
        y_pred = transform1(y_pred)
        zero = torch.zeros_like(y_pred)
        one = torch.ones_like(y_pred)
        y_pred = torch.where(y_pred > 0.5, one, zero)
        y_pred = y_pred.numpy().squeeze()

        y_pred = y_pred.astype('float32')

        e, f, g, j, precision, specificity, sensitivity,accuary,f1,iou = sorensen_dices(y_true, y_pred)
        DSIs = DSIs + e
        FPDs = FPDs + f
        FNDs = FNDs + g
        Jss = Jss + j
        precisions = precisions + precision
        specificitys = specificitys + specificity
        sensitivitys = sensitivitys + sensitivity
        accuarys=accuarys+accuary
        f1s = f1s + f1
        ious = ious+iou
    DSIsm = DSIs / sum
    FPDsm = FPDs / sum
    FNDsm = FNDs / sum
    Jssm = Jss / sum
    precisionsm = precisions / sum
    specificitysm = specificitys / sum
    sensitivitysm = sensitivitys / sum
    accuarysm = accuarys / sum
    f1sm = f1s / sum
    iousm = ious/sum
    print(DSIsm, FPDsm, FNDsm, precisionsm, Jssm, specificitysm, sensitivitysm,accuarysm,f1sm,iousm)

    test_txt = "/opt/data/private/log/sar/test2_c.txt"
    os.makedirs(os.path.dirname(test_txt), exist_ok=True)
    output = "%s:DSIsm:%f,FPDsm:%f,FNDsm:%f,precisionsm:%f,Jssm:%f,specificitysm:%f,sensitivitysm:%f,accuarysm:%f,f1sm:%f,iousm:%f" % (datetime.now(), DSIsm, FPDsm, FNDsm,precisionsm, Jssm, specificitysm,sensitivitysm,accuarysm,f1sm,iousm)
    with open(test_txt, "a+") as f:
        f.write(output + '\n')
        f.close
    # Js方差
    return

# test()
Evaluation(Predict_Path=r'/opt/data/private/dataset/CVC dataset/CVC dataset/result/c2/', Label_Path=r'/opt/data/private/dataset/CVC dataset/CVC dataset/result_gt/c2/', target_size=(256,256), Begin=1, End=153)
