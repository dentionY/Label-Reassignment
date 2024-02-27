import os
import numpy as np
import shutil


rootdir = './UAV-benchmark-M'  # 源数据集图像的文件夹的路径
traindir = './UAV-benchmark-M-train'
valdir = './UAV-benchmark-M-val'
testdir = './UAV-benchmark-M-test'

path = os.listdir(rootdir)
np.random.shuffle(path)  # 将数据集打乱顺序
print("path is ", path)


d1 = 37  
d2 = 40
train = path[:d1]  
val = path[d1:d2]  
test = path[d2:]

for i in train:
    ori_path = rootdir + '/' + str(i)
    train_path = traindir + '/' + str(i)
    shutil.copytree(ori_path,train_path)

for i in val:
    ori_path = rootdir + '/' + str(i)
    val_path = valdir + '/' + str(i)
    shutil.copytree(ori_path,val_path)

for i in test:
    ori_path = rootdir + '/' + str(i)
    test_path = testdir + '/' + str(i)
    shutil.copytree(ori_path,test_path)
    
'''
输出内容：
(base) root@autodl-container-c8da118cfa-3261e683:~/autodl-tmp/uavdt# python ShuffleEpisode.py 
path is  ['M1001', 'M1007', 'M1003', 'M0802', 'M0301', 'M0208', 'M1401', 'M0401', 'M0206', 'M1008', 'M1009', 'M0603', 'M0202', 'M0201', 'M1306', 'M0604', 'M0601', 'M0205', 'M0702', 'M0701', 'M1005', 'M1102', 'M0602', 'M0209', 'M1004', 'M1202', 'M0204', 'M1301', 'M1201', 'M0704', 'M0203', 'M1101', 'M0605', 'M0403', 'M0703', 'M0210', 'M0101', 'M1305', 'M1302', 'M0902', 'M1304', 'M0901', 'M0606', 'M1303', 'M1006', 'M1002', 'M0402', 'M0207', 'M0801', 'M0501']
'''