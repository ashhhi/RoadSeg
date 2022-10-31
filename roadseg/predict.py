import glob
import numpy as np
import torch
import os
import cv2
from utils.utils import  FWIoU
from skimage import io
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
transform1 = A.Compose([
    A.Resize(256, 256),
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class seg_qyl(nn.Module):
    def __init__(self, model_name, n_class):
        super().__init__()
        self.model = smp.UnetPlusPlus(# UnetPlusPlus
                encoder_name=model_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=n_class,                      # model output channels (number of classes in your dataset)
            )
    def forward(self, x):
        #with autocast():
        x = self.model(x)
        return x
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    # 加载网络，图片单通道，分类为1。
    # net = Net(3, 1)
    model_name = 'xception'  # xception
    n_class = 1
    net = seg_qyl(model_name, n_class)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('outputs/crop/xception/checkpoint-epoch182_fwiou99.67.pth')['state_dict'])
    # net.load_state_dict(torch.load('cpts/e199_200_s256-b7.pth', map_location=device))
    # torch.save(net.state_dict(),'checkpoints/SAR/a_new_MPnet.pth', _use_new_zipfile_serialization=False)
    # net.load_state_dict(torch.load('checkpoints/XT/best_model.pth', map_location=device))
    #
    # 重新保存网络参数，此时注意改为非zip格式
    #torch.save(net.state_dict(), 'checkpoints/XT/aMPResnet.pth', _use_new_zipfile_serialization=False)
    # # 测试模式
    net.eval()

    # model_name2 = 'resnet101'  # xception
    # net2 = seg_qyl(model_name2, n_class).to(device=device)
    # net2.load_state_dict(torch.load('Unetpll_resnet101.pth', map_location=device))
    # net2.eval()
    #
    # model = Net(1, 1).to(device=device)
    # model.load_state_dict(torch.load('bestMPResnet.pth', map_location=device))
    # model.eval()
    # # 读取所有图片路径
    # data_dir = r'D:\SAR/Sarsegment/datasets/'
    tests_path = glob.glob(r'D:\SAR/Sarsegment/datasets/val/image/*.png')
    # tests_path = glob.glob('dataset/val/image/*.png')
    print(len(tests_path))
    # 遍历素有图片
    res_iou = []
    with torch.no_grad():
        for test_path in tests_path:
            # 保存结果地址
            save_res_path = test_path.split('.')[0] + '_res.png'
            save_res_path = save_res_path.replace('image', 'result')
            # if not os.path.exists(save_res_path):
            #     os.makedirs(save_res_path)
            # 读取图片
            # img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(test_path)
            t1, t2 = img.shape[0], img.shape[1]
            label_path = test_path.replace('image', 'gt')
            label = cv2.imread(test_path.replace('image', 'gt'), cv2.IMREAD_GRAYSCALE) / 255
            # if np.mean(label) != 0:
            # label = label[0:t1 // 2, 0:t2 // 2]
            img1 = img[0:t2//2, 0:t1//2, ...]
            img2 = img[t2//2:t2, 0:t1//2, ...]
            img3 = img[0:t2//2, t1//2:t1, ...]
            img4 = img[t2//2:t2, t1//2:t1, ...]
            #img = cv2.resize(img, (256, 256))

            transformed1 = transform1(image=img1, mask=label)
            img1 = transformed1['image'] / 1
            img1 = img1.reshape(1, 3, 256, 256).to(device)
            pred1 = net(img1)
            pred1 = torch.sigmoid(pred1).cpu().data.numpy()[0, 0]

            transform2 = A.Compose([
                A.Resize(t2//2, t1//2),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            transformed2 = transform2(image=pred1, mask=label)
            pred1 = transformed2['image']

            transformed1 = transform1(image=img2, mask=label)
            img2 = transformed1['image'] / 1
            img2 = img2.reshape(1, 3, 256, 256).to(device)
            pred2 = net(img2)
            pred2 = torch.sigmoid(pred2).cpu().data.numpy()[0, 0]
            transformed2 = transform2(image=pred2, mask=label)
            pred2 = transformed2['image']

            transformed1 = transform1(image=img3, mask=label)
            img3 = transformed1['image'] / 1
            img3 = img3.reshape(1, 3, 256, 256).to(device)
            pred3 = net(img3)
            pred3 = torch.sigmoid(pred3).cpu().data.numpy()[0, 0]
            transformed2 = transform2(image=pred3, mask=label)
            pred3 = transformed2['image']

            transformed1 = transform1(image=img4, mask=label)
            img4 = transformed1['image'] / 1
            img4 = img4.reshape(1, 3, 256, 256).to(device)
            pred4 = net(img4)
            pred4 = torch.sigmoid(pred4).cpu().data.numpy()[0, 0]
            transformed2 = transform2(image=pred4, mask=label)
            pred4 = transformed2['image']

            con1 = np.concatenate((pred1,pred3),axis=1)
            con2 = np.concatenate((pred2,pred4),axis=1)
            pred = np.concatenate((con1,con2),axis=0).squeeze()
            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            # cv2.imshow('pred1', con1)
            # cv2.imshow('pred2', con2)
            # cv2.imshow('pred', pred)
            # cv2.imshow('gt', label)
            # cv2.waitKey()
            # print(pred.shape, label.shape)
            fwiou = FWIoU(pred, label, ignore_zero=True)
            res_iou.append(fwiou)
            # transform2 = A.Compose([
            #     A.Resize(t2, t1),
            #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ])
            # transformed2 = transform2(image=pred, mask=label)
            # pred = transformed2['image']
            # transformed1 = transform1(image=img, mask=label)
            # img = transformed1['image'] / 1
            # img = img.reshape(1, 3, 256, 256).to(device)
            # transformed1 = transform1(image=img, mask=label)
            # img = transformed1['image'] / 1
            # pred = net(img)
            # pred = torch.sigmoid(pred).cpu().data.numpy()[0, 0]
            # fwiou = FWIoU(pred.squeeze(), label.squeeze(), ignore_zero=True)
            # transform2 = A.Compose([
            #     A.Resize(t2, t1),
            #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ])
            # transformed2 = transform2(image=pred, mask=label)

            #label = cv2.resize(label, (256, 256))
            #img = img.transpose(2,0,1)
            #print(img.shape)
            # img = img.reshape(1, 3, 256, 256).to(device)
            # # 转为tensor
            # #img = torch.from_numpy(img)
            # # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            # #img = img.to(device=device, dtype=torch.float32)
            #
            # # image_flip2 = torch.flip(img, [2])
            # # image_flip3 = torch.flip(img, [3])
            # # img_rot90 = torch.rot90(img, 1, dims=(2, 3))
            # # 预测
            # #print(img_tensor.shape)
            # pred = net(img)
            # # pred_flip2 = net(image_flip2)
            # # pred_flip3 = net(image_flip3)
            # # pred_rot90 = net(img_rot90)
            # # pred_flip2 = torch.flip(pred_flip2, [2])
            # # pred_flip3 = torch.flip(pred_flip3, [3])
            # # pred_rot90 = torch.rot90(pred_rot90, -1, dims=(2, 3))
            # pred = torch.sigmoid(pred).cpu().data.numpy()[0, 0]
            # label = label.cpu().data.numpy()
            # transform2 = A.Compose([
            #     A.Resize(t2, t1),
            #     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ])
            # transformed2 = transform2(image=pred, mask=label)
            # pred = transformed2['image']

            # pred_flip2 = torch.sigmoid(pred_flip2).cpu().data.numpy()[0, 0]
            # pred_flip3 = torch.sigmoid(pred_flip3).cpu().data.numpy()[0, 0]
            # pred_rot90 = torch.sigmoid(pred_rot90).cpu().data.numpy()[0, 0]
            # # 提取结果
            # first_pred = (pred + pred_flip2 + pred_flip3 + pred_rot90) / 4.0
            # label = np.array(label.data.cpu())
            # 处理结果
            # img = cv2.imread(test_path)
            # t1, t2 = img.shape[0], img.shape[1]
            # img = cv2.resize(img, (512, 512))
            # label_path = test_path.replace('image', 'gt')
            # label = cv2.imread(test_path.replace('image', 'gt'), cv2.IMREAD_GRAYSCALE) / 255
            #
            # img = img.transpose(2, 0, 1)
            # # print(img.shape)
            # img = img.reshape(1, 3, 512, 512)
            # # 转为tensor
            # img = torch.from_numpy(img)
            # # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            # img = img.to(device=device, dtype=torch.float32)
            # image_flip2 = torch.flip(img, [2])
            # image_flip3 = torch.flip(img, [3])
            # img_rot90 = torch.rot90(img, 1, dims=(2, 3))
            # # 预测
            # # print(img_tensor.shape)
            # pred = net2(img)
            # pred_flip2 = net2(image_flip2)
            # pred_flip3 = net2(image_flip3)
            # pred_rot90 = net2(img_rot90)
            # pred_flip2 = torch.flip(pred_flip2, [2])
            # pred_flip3 = torch.flip(pred_flip3, [3])
            # pred_rot90 = torch.rot90(pred_rot90, -1, dims=(2, 3))
            # pred = torch.sigmoid(pred).cpu().data.numpy()[0, 0]
            # pred_flip2 = torch.sigmoid(pred_flip2).cpu().data.numpy()[0, 0]
            # pred_flip3 = torch.sigmoid(pred_flip3).cpu().data.numpy()[0, 0]
            # pred_rot90 = torch.sigmoid(pred_rot90).cpu().data.numpy()[0, 0]
            # # 提取结果
            # third_pred = (pred + pred_flip2 + pred_flip3 + pred_rot90) / 4.0
            #
            # img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
            # t1, t2 = img.shape[0], img.shape[1]
            # img = cv2.resize(img, (512, 512))
            # # img = img.transpose(2, 0, 1)
            # # print(img.shape)
            # img = img.reshape(1, 1, 512, 512)
            # # 转为tensor
            # img = torch.from_numpy(img)
            # # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            # img = img.to(device=device, dtype=torch.float32)
            # image_flip2 = torch.flip(img, [2])
            # image_flip3 = torch.flip(img, [3])
            # img_rot90 = torch.rot90(img, 1, dims=(2, 3))
            # # 预测
            # # print(img_tensor.shape)
            # pred, _ = model(img)
            # pred_flip2, _ = model(image_flip2)
            # pred_flip3, _ = model(image_flip3)
            # pred_rot90, _ = model(img_rot90)
            # pred_flip2 = torch.flip(pred_flip2, [2])
            # pred_flip3 = torch.flip(pred_flip3, [3])
            # pred_rot90 = torch.rot90(pred_rot90, -1, dims=(2, 3))
            # pred = torch.sigmoid(pred).cpu().data.numpy()[0, 0]
            # pred_flip2 = torch.sigmoid(pred_flip2).cpu().data.numpy()[0, 0]
            # pred_flip3 = torch.sigmoid(pred_flip3).cpu().data.numpy()[0, 0]
            # pred_rot90 = torch.sigmoid(pred_rot90).cpu().data.numpy()[0, 0]
            # # 提取结果
            # second_pred = (pred + pred_flip2 + pred_flip3 + pred_rot90) / 4.0

            # pred = first_pred * 0.3 + second_pred * 0.4 + second_pred * 0.3

            # pred = cv2.resize(pred, (t2, t1))
            # pred[pred > 0.5] = 1
            # pred[pred <= 0.5] = 0
            #
            # fwiou = FWIoU(pred.squeeze(), label.squeeze(), ignore_zero=True)
            # res_iou.append(fwiou)
            if fwiou < 0.98:
                print(test_path, fwiou, t1, t2)
            # 保存图片
            cv2.imwrite(save_res_path, pred * 255)
        print('avefw:' + str(np.mean(res_iou)))
