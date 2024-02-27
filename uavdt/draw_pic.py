
# coding:utf-8
# 检查json格式转换的是否正确（即，将转换的数据画回原图中）
# 可以img show简单查看。
from pycocotools.coco import COCO
import cv2
import pandas as pd
import json


def select(json_path, outpath, image_path):
    json_file = open(json_path)
    infos = json.load(json_file)
    images = infos["images"]
    annos = infos["annotations"]
    assert len(images) == len(images)
    for i in range(len(images)):
        im_id = images[i]["id"]
        im_path = image_path + images[i]["file_name"]
        print(im_path)
        img = cv2.imread(im_path)
        for j in range(len(annos)):
            # import pdb
            # pdb.set_trace()
            if annos[j]["image_id"] == im_id:
                # import pdb
                # pdb.set_trace()
                x, y, w, h = annos[j]["bbox"]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x2, y2 = x + w, y + h
                # object_name = annos[j][""]
                img = cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), thickness=2)
                img_name = outpath + images[i]["file_name"]
                # cv2.imshow('uavdt',img)
                # cv2.waitKey(0)
                cv2.imwrite(img_name, img)
                # continue
        cv2.imshow('uavdt',img)
        cv2.waitKey(0)

        # print(i)

if __name__ == "__main__":
    # imageidFile = ''
    json_path = './UAV-new.json'
    image_path = './testjsonimage/'
    outpath = './outtestjsonimage/'
    select(json_path, outpath, image_path)