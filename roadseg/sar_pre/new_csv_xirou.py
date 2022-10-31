import csv
import os

path1 = '/home/dxy/wmw/data/Kvasir-Capsule/images/'  # 所需修改文件夹所在路径
dirs1 = os.listdir(path1)

path2 = '/home/dxy/wmw/data/Kvasir-Capsule/masks/'
dirs2 = os.listdir(path2)

with open("metadata3.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_id", "sat_image_path", "mask_path"])
    j = 0
    for dir in dirs1:
        for dir0 in dirs2:
            #若原图与标签名称一致则断定为对应组图
            if dir==dir0:
                writer.writerow([j, 'image/'+dir, 'gt/'+dir0])
            j=j+1
