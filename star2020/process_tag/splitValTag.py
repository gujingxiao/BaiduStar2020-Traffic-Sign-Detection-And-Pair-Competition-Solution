# -*- coding: utf-8 -*-
"""
@author: Jingxiao Gu
"""
import os
import json
import shutil
import random

labels = {
    '102': 0,
    '103': 1,
    '104': 2,
    '105': 3,
    '106': 4,
    '107': 5,
    '108': 6,
    '109': 7,
    '110': 8,
    '111': 9,
    '112': 10,
    '201': 11,
    '202': 12,
    '203': 13,
    '204': 14,
    '205': 15,
    '206': 16,
    '207': 17,
    '301': 18,
}


# Split train and val
if __name__ == '__main__':
    tagFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/total_tag/"
    trainFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/train_tag/"
    valFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/val_tag/"

    if os.path.exists(trainFolder):
        os.removedirs(trainFolder)
    os.mkdir(trainFolder)

    if os.path.exists(valFolder):
        os.removedirs(valFolder)
    os.mkdir(valFolder)

    train_list = []
    train_count = [0] * 19
    val_list = []
    val_count = [0] * 19

    total_list = os.listdir(tagFolder)
    random.shuffle(total_list)
    val_length = max(1, int(0.1 * len(total_list)))

    train_list = total_list[val_length:]
    val_list = total_list[0:val_length]

    for tagFile in train_list:
        tagPath = os.path.join(tagFolder, tagFile)

        fJson = open(tagPath)
        gtJson = json.load(fJson)
        gtSigns = gtJson['signs']

        for gtSign in gtSigns:
            train_count[labels[gtSign['type']]] += 1

        shutil.copy(tagPath, os.path.join(trainFolder, tagFile))

    for tagFile in val_list:
        tagPath = os.path.join(tagFolder, tagFile)

        fJson = open(tagPath)
        gtJson = json.load(fJson)
        gtSigns = gtJson['signs']

        for gtSign in gtSigns:
            val_count[labels[gtSign['type']]] += 1

        shutil.copy(tagPath, os.path.join(valFolder, tagFile))

    print(train_count)
    print(val_count)