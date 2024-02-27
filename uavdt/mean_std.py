import numpy as np
import cv2
import os
 

#img_h, img_w = 32, 32
'''
normMean = [0.37277249, 0.38563663, 0.38452408]
normStd = [0.20723985, 0.20523821, 0.21362634]
'''
img_h, img_w = 32, 48   #根据自己数据集适当调整，影响不大
'''
normMean = [0.37225845, 0.38512826, 0.38392654]
normStd = [0.20772879, 0.20576294, 0.21414135]
'''
means, stdevs = [], []
img_list = []
 
imgs_path = './UAV-new'
imgs_path_list = os.listdir(imgs_path)
 
len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)    
 
imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))