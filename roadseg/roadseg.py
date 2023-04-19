import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import segmentation_models_pytorch as smp
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
import albumentations as album
from attributes.setting import DATA_DIR, LABEL_DIR, TRAINING, PRED_DIR

class RoadsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, label_paths, class_rgb_values=None, augmentation=None, preprocessing=None, ):
        self.image_paths = image_paths
        self.mask_paths = label_paths
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read images and masks
        image = cv2.imread(self.image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        return image, mask

    def __len__(self):
        # return length of
        return len(self.image_paths)

# album.Resize((256, 256)),
def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.Resize(256, 256),
    ]
    return album.Compose(train_transform)


def get_valid_augmentation():
    valid_transform = [
        album.Resize(256, 256),
    ]
    return album.Compose(valid_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')   # transpose转置，0.1.2分别表示xyz轴


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        # albumentations 是一个数据增强工具
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


if __name__ == '__main__':
    metadata = []
    metalabel = []
    for filename in os.listdir(DATA_DIR):
        name = filename.split('.')[0]
        for filename1 in os.listdir(LABEL_DIR):
            if filename1.split('.')[0] == name:
                metadata.append(os.path.join(DATA_DIR,filename))
                metalabel.append(os.path.join(LABEL_DIR,filename1))

    class_dict = pd.read_csv(r'attributes/class_dict.csv')
    # Get class names  sd
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

    select_classes = ['background', 'road']
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    # 九比一划分训练集和测试集
    index = int(len(metadata) / 10)
    train = metadata[index:]
    train_label = metalabel[index:]
    valid = metadata[:index]
    valid_label = metalabel[:index]

    ENCODER = 'efficientnet-b7'  # resnet50,
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = select_classes
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

    # 用预训练好的encoder创建分割模型
    model = smp.DeepLabV3Plus(  #Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
        encoder_name=ENCODER,   # 选择解码器
        encoder_weights=ENCODER_WEIGHTS,    # 使用预先训练的权重imagenet进行解码器初始化
        classes=len(CLASSES),   # 模型输出通道（数据集所分的类别总数）
        activation=ACTIVATION,
    )

    # 所有模型均具有预训练的编码器，因此必须按照权重预训练的相同方法准备数据
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained=ENCODER_WEIGHTS)

    # 加载训练数据集
    train_dataset = RoadsDataset(image_paths=train,
                                 label_paths=train_label,
                                 augmentation=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn),
                                 class_rgb_values=select_class_rgb_values,
                                 )
    # 加载验证数据集
    valid_dataset = RoadsDataset(image_paths=valid,
                                 label_paths=valid_label,
                                 augmentation=get_valid_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn),
                                 class_rgb_values=select_class_rgb_values,
                                 )

    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True)
    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)

    # Set num of epochs
    EPOCHS = 40

    # Set device: `cuda` or `cpu`
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Accuracy(),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.Precision(),
        smp.utils.metrics.Recall(),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    valid_logs = None
    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists('saved/best_model.pth'):
        model = torch.load('saved/best_model.pth',
                           map_location=DEVICE)
        valid_logs = torch.load('saved/best_valid_logs.pth',
                           map_location=DEVICE)
        print('Loaded pre-trained DeepLabV3+ model!')

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )



    if TRAINING:
        if valid_logs != None:
            best_iou_score = valid_logs['iou_score']
        else:
            best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):
            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, 'saved/best_model.pth')
                torch.save(valid_logs, 'saved/best_valid_logs.pth')
                print('Model saved!')
    else:
        # load best saved model checkpoint from the current run
        if os.path.exists('saved/best_model.pth'):
            best_model = torch.load('./saved/best_model.pth', map_location=DEVICE)
            print('Loaded DeepLabV3+ model from this run.')
        else:
            assert False, 'No Model!'

        test_dataset = RoadsDataset(
            image_paths=metadata,
            label_paths=metalabel,
            augmentation=get_valid_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            class_rgb_values=select_class_rgb_values,
        )
        # test dataset for visualization (without preprocessing augmentations & transformations)
        test_dataset_vis = RoadsDataset(
            image_paths=metadata,
            label_paths=metalabel,
            augmentation=get_valid_augmentation(),
            class_rgb_values=select_class_rgb_values,
        )


        num = 0
        while True:
            if not os.path.exists(os.path.join(PRED_DIR,f'exp{num}')):
                os.makedirs(os.path.join(PRED_DIR,f'exp{num}'))
                break
            else:
                num += 1
        for idx in range(len(test_dataset)):
            print("starting test:", idx)
            image, gt_mask = test_dataset[idx]
            image_vis = test_dataset_vis[idx][0].astype('uint8')
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
            # Predict test image
            pred_mask = best_model(x_tensor)
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
            # Convert pred_mask from `CHW` format to `HWC` format
            pred_mask = np.transpose(pred_mask, (1, 2, 0))
            # Get prediction channel corresponding to foreground
            pred_road_heatmap = pred_mask[:, :, select_classes.index('road')]
            pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values)
            # Convert gt_mask from `CHW` format to `HWC` format
            gt_mask = np.transpose(gt_mask, (1, 2, 0))
            gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
            cv2.imwrite(os.path.join(PRED_DIR,f"exp{num}/pred_{idx}.png"),np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])




