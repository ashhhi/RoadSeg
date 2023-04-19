import os
import shutil

path = r'C:\DataSet\deepglobe-road-dataset\train'
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
for file in files:
    file_type = file.split('_')[1].split('.')[0]
    file_name = file.split('_')[0]
    new_name = file_name + '.png'
    fullpath = os.path.join(path, new_name)
    os.rename(os.path.join(path, file), fullpath)

    if file_type == 'mask':
        mymovefile(fullpath, gt_path)
    else:
        mymovefile(fullpath, train_path)


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
            os.rename(path, os.path.join(file_dir, new_name))
        else:
            # print("{0} : is dir!".format(cur_file))
            list_dir(path) # 递归子目录