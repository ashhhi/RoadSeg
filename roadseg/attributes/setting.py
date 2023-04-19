import os.path

# 开启训练
TRAINING = True

# 数据路径
ROOT_DIR = r'C:\DataSet\deepglobe-road-dataset\train'
DATA_DIR = os.path.join(ROOT_DIR, 'train')
LABEL_DIR = os.path.join(ROOT_DIR, 'gt')
PRED_DIR = os.path.join(ROOT_DIR, 'pred')
