import cv2.cv2 as cv
import cv2
import os
import shutil

# images_dir='/media/limzero/compete_datasets/suichang_round1_train_210120/'
save_imgs='/data1/wmw/dataset/weixing/test/'
save_masks='/data1/wmw/dataset/weixing/gt/'
if not os.path.exists(save_imgs):os.makedirs(save_imgs)
if not os.path.exists(save_masks):os.makedirs(save_masks)

path = '/data1/wmw/dataset/weixing/data/'  # 所需修改文件夹所在路径
dirs = os.listdir(path)

for dir in dirs:
    path2 = str(path + dir + '/')
    print(path2)
    dir0 = os.listdir(path2)
    # print(dir0)
    for i in dir0:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(i)[1] == ".tif":
            if i.split('_')[1] == "mask":  # 筛选文件
                img = cv.imread(os.path.join(path2, i), cv2.IMREAD_GRAYSCALE)  # 读取列表中的tif图像
                retval, img = cv.threshold(img, 1, 255, cv2.THRESH_BINARY)
                cv.imwrite(os.path.join(save_imgs, i.split('.')[0] + ".png"), img)  # tif 格式转 jpg

                print("read mask!")
            else:
                img = cv.imread(os.path.join(path2, i), -1)  # 读取列表中的tif图像
                cv.imwrite(os.path.join(save_imgs, i.split('.')[0] + ".jpg"), img)  # tif 格式转 jpg
                print("read img!")


