import csv
import os

path1 = '/data1/wmw/dataset/weixing/test/'  # 所需修改文件夹所在路径
dirs1 = os.listdir(path1)

path2 = '/data1/wmw/dataset/weixing/gt/'
dirs2 = os.listdir(path2)

with open("metadata.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_id", "sat_image_path", "mask_path"])
    j = 0
    for dir in dirs1:
        for dir0 in dirs2:
            if dir.split('_')[0]==dir0.split('_')[0]:
                writer.writerow([j, 'test/'+dir, 'gt/'+dir0])
            j=j+1
