import os
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim
import cv2
from glob import glob
from PIL import Image
from tqdm import tqdm
from utils import eval_segm as seg_acc
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pytorch_toolbelt import losses as L
from utils import AverageMeter, second2time, inial_logger
from albumentations.augmentations import functional as F
from .metric import IOUMetric
from torch.cuda.amp import autocast, GradScaler  # need pytorch>1.6
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, SoftCrossEntropyLoss, LovaszLoss, \
    SoftBCEWithLogitsLoss
from torch.nn import BCEWithLogitsLoss
from torch.optim.swa_utils import AveragedModel, SWALR

Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def binary_accuracy(pred, label):
    valid = (label < 2)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def FWIoU(pred, label, bn_mode=False, ignore_zero=False):
    if bn_mode:
        pred = (pred >= 0.5)
        label = (label >= 0.5)
    elif ignore_zero:
        pred = pred - 1
        label = label - 1
    FWIoU = seg_acc.frequency_weighted_IU(pred, label)
    return FWIoU


def train_net(param, model, train_data, valid_data, plot=False, device='cuda'):
    # 初始化参数
    model_name = param['model_name']
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']

    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    min_inter = param['min_inter']
    iter_inter = param['iter_inter']

    save_log_dir = param['save_log_dir']
    save_ckpt_dir = param['save_ckpt_dir']
    load_ckpt_dir = param['load_ckpt_dir']
    #
    scaler = GradScaler()

    # swa_model = AveragedModel(model).to('cuda')
    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5,
                                                                     last_epoch=-1)
    # swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    DiceLoss_fn = DiceLoss(mode='binary')
    bceLoss_fn = BCEWithLogitsLoss()
    # SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    criterion = L.JointLoss(first=DiceLoss_fn, second=bceLoss_fn,
                            first_weight=0.5, second_weight=0.5).cuda()
    # criterion = LovaszLoss(mode='binary')
    # criterion = BCEWithLogitsLoss()
    # criterion = SoftBCEWithLogitsLoss()
    logger = inial_logger(
        os.path.join(save_log_dir, time.strftime("%m-%d %H_%M_%S", time.localtime()) + '_' + model_name + '.log'))
    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_iou = 0
    best_epoch = 0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info(
        'Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size,
                                                                                       valid_data_size))
    #
    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        # acc_meter = AverageMeter()
        train_main_loss = AverageMeter()
        curr_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            # print(data.dtype, target.dtype)
            target = target.unsqueeze(1)
            # curr_iter = curr_epoch * len(train_loader)
            # running_iter = curr_iter + batch_idx + 1
            # adjust_lr(optimizer, running_iter, all_iters)
            with autocast():  # need pytorch>1.6

                pred = model(data)
                loss = criterion(pred, target)
                # print(pred.shape, target.shape)
                # pred, aux = model(data)
                # main_loss = criterion(pred, target)
                # aux_loss = criterion(aux, target)
                # loss = main_loss * 0.7 + aux_loss * 0.3

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            scheduler.step(epoch + batch_idx / train_loader_size)

            train_main_loss.update(loss.cpu().detach().numpy())
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    optimizer.param_groups[-1]['lr'],
                    train_main_loss.avg, spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                # train_iter_loss.reset()
        # if epoch > swa_start_epoch:
        #     swa_model.update_parameters(model)
        #     swa_scheduler.step()
        #     print('saw-started')
        # else:
        #     scheduler.step()
        # 验证阶段
        model.eval()
        val_loss = AverageMeter()
        acc_meter = AverageMeter()
        fwIoU_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                target = target.unsqueeze(1)
                pred = model(data).float()
                loss = criterion(pred, target)
                # pred, aux = model(data)
                # main_loss = criterion(pred, target)
                # aux_loss = criterion(aux, target)
                # loss = main_loss * 0.7 + aux_loss * 0.3
                pred = torch.sigmoid(pred)
                pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
                pred = pred.cpu().data.numpy()
                # pred= np.argmax(pred,axis=1)
                # print(pred.mean(), target.mean().item())
                # iou.add_batch(pred,target.cpu().data.numpy())
                # #
                image_loss = loss.item()
                # valid_epoch_loss.update(image_loss)
                # valid_iter_loss.update(image_loss)
                val_loss.update(loss.cpu().detach().numpy())
                if batch_idx % iter_inter == 0:
                    logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                        epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, val_loss.avg))
                # val_loss=valid_iter_loss.avg
                # acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()

                for (pred, label) in zip(pred, target):
                    pred_512 = cv2.resize(pred.squeeze(), (512, 512))
                    label_512 = cv2.resize(label.cpu().data.numpy().squeeze(), (512, 512))
                    acc, valid_sum = binary_accuracy(pred, label)
                    fwiou = FWIoU(pred_512, label_512, ignore_zero=True)
                    # fwiou = FWIoU(pred.squeeze(), label.cpu().squeeze(), ignore_zero=True)
                    acc_meter.update(acc)
                    fwIoU_meter.update(fwiou)
            # logger.info('[val] epoch:{} miou:{:.2f}'.format(epoch,mean_iu))

        # 保存loss、lr
        train_loss_total_epochs.append(train_main_loss.avg)
        valid_loss_total_epochs.append(val_loss.average())
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        # if epoch % save_inter == 0 and epoch > min_inter:
        #     state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        #     filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        #     torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if fwIoU_meter.average() > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            filename = os.path.join(save_ckpt_dir,
                                    'checkpoint-epoch{}_fwiou{:.2f}.pth'.format(epoch, fwIoU_meter.average() * 100))
            torch.save(model.state_dict(), filename, _use_new_zipfile_serialization=False)
            best_iou = fwIoU_meter.average()
            best_mode = copy.deepcopy(model)
            best_epoch = epoch
            logger.info(
                '[save] Best Model saved at epoch:{}， fwiou:{} ============================='.format(epoch, best_iou))
        # scheduler.step()
        # 显示loss
        print('best_epoch:', best_epoch, 'bestIoU:', best_iou * 100, 'nowIoU:', fwIoU_meter.average() * 100)

    # torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    # torch.save(swa_model.state_dict(), os.path.join(save_ckpt_dir, 'unetppl_b7_swa_model.pth'.format(epoch)))
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr, label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()

    return best_mode, model
#
