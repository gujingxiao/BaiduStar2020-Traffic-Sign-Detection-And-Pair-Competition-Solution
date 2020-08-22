# -*- coding: utf-8 -*-
"""
Created on Tues Jun 30 10:20:01 2020

@author: Jingxiao Gu
"""

import cv2
import json
import numpy as np
import os


labels = {
    '102': 102,
    '103': 102,
    '104': 102,
    '105': 102,
    '106': 102,
    '107': 102,
    '108': 102,
    '109': 102,
    '110': 102,
    '111': 102,
    '112': 102,
    '201': 103,
    '202': 103,
    '203': 103,
    '204': 103,
    '205': 103,
    '206': 103,
    '207': 103,
    '301': 104,
}

def parseJson(tagPath, savePath):
    fJson = open(tagPath, encoding="utf-8")
    tagJson = json.load(fJson)

    for sign in tagJson['signs']:
        label = sign['type']
        sign['type'] = str(labels[label])

    #print(tagJson)
    with open(savePath, "w") as f:
        json.dump(tagJson, f)


if __name__ == '__main__':
    baseFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/val_tag"
    saveFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/val_tag3"

    for index, tag in enumerate(os.listdir(baseFolder)):
        tagPath = os.path.join(baseFolder, tag)
        parseJson(tagPath, os.path.join(saveFolder, tag))
        if index % 100 == 0:
            print(index)