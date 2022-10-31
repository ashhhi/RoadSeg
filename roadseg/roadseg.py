import os
import cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import segmentation_models_pytorch as smp
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album

class RoadsDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_rgb_values=None, augmentation=None, preprocessing=None, ):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read images and masks
        #print(self.image_paths[i])
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)

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
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(_transform)


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.savefig('squares_plot.png', bbox_inches='tight')


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
    print('test8')
    DATA_DIR = '/home/dxy/wmw/data/Kvasir dataset/train'
    metadata_df = pd.read_csv('/home/dxy/wmw/code/sar/sar_pre/metadata_kvasir.csv')
    # print(metadata_df)
   # metadata_df = metadata_df[metadata_df['split'] == 'train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    # Shuffle DataFrame
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    # print(metadata_df)

    class_dict = pd.read_csv('/home/dxy/wmw/code/sar/class_dict.csv')
    # Get class names  sd
    class_names = class_dict['name'].tolist()
    # Get class RGB values
    class_rgb_values = class_dict[['r', 'g', 'b']].values.tolist()

    select_classes = ['background', 'road']
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    # Perform 90/10 split for train / val
    valid_df = metadata_df.sample(frac=0.1, random_state=42)
    train_df = metadata_df.drop(valid_df.index)

    print(valid_df)

    ENCODER = 'efficientnet-b7'  # resnet50,
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = select_classes
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.UnetPlusPlus(  #Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = RoadsDataset(train_df,
                                 augmentation=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn),
                                 class_rgb_values=select_class_rgb_values,
                                 )

    valid_dataset = RoadsDataset(valid_df,
                                 augmentation=get_valid_augmentation(),
                                 preprocessing=get_preprocessing(preprocessing_fn),
                                 class_rgb_values=select_class_rgb_values,
                                 )
    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

    # Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
    TRAINING = True

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
        dict(params=model.parameters(), lr=0.00008),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=5e-5,
    )

    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists('outputs/best_model.pth'):
        model = torch.load('outputs/best_model.pth',
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
        best_iou_score = 0.0
        train_logs_list, valid_logs_list = [], []

        for i in range(0, EPOCHS):

            # Perform training & validation
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            print("train IOU is:", train_logs['iou_score'])
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            # Save model if a better val IoU score is obtained
            print("valid IOU,acc,f1,precision,recall is:", valid_logs['iou_score'])
            if best_iou_score < valid_logs['iou_score']:
                best_iou_score = valid_logs['iou_score']
                torch.save(model, './best_model_k_1.pth')
                print('Model saved!')
    else:
        # load best saved model checkpoint from the current run
        if os.path.exists('./best_model.pth'):
            best_model = torch.load('./best_model.pth', map_location=DEVICE)
            print('Loaded DeepLabV3+ model from this run.')

        test_dataset = RoadsDataset(
            metadata_df,
            augmentation=get_valid_augmentation(),
            preprocessing=get_preprocessing(preprocessing_fn),
            class_rgb_values=select_class_rgb_values,
        )
        # test dataset for visualization (without preprocessing augmentations & transformations)
        test_dataset_vis = RoadsDataset(
            metadata_df,
            augmentation=get_valid_augmentation(),
            class_rgb_values=select_class_rgb_values,
        )

        sample_preds_folder = '/data1/wmw/dataset/pre/'
        if not os.path.exists(sample_preds_folder):
            os.makedirs(sample_preds_folder)

        print(len(test_dataset))

        for idx in range(len(test_dataset)):
            print("starting test ", idx)
            image, gt_mask = test_dataset[idx]
            print(gt_mask)
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
            print(gt_mask)
            gt_mask = colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values)
            print(gt_mask)
            cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"),
                        np.hstack([image_vis, gt_mask, pred_mask])[:, :, ::-1])




