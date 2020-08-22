# -*- coding: utf-8 -*-
"""
@author: Jingxiao Gu
"""
import os
import cv2
import json

gtFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test/input/"
imageFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test/pic/"

imageResolutions = {}
for index, tag in enumerate(os.listdir(gtFolder)):
    gtPath = os.path.join(gtFolder, tag)
    fJson = open(gtPath)
    gtJson = json.load(fJson)
    for pic in gtJson['group'][0]['pic_list']:
        image = cv2.imread(os.path.join(imageFolder, pic + '.jpg'))
        print(index, pic, image.shape[1], image.shape[0])
        imageResolutions[pic] = [image.shape[1], image.shape[0]]
    for pic in gtJson['group'][1]['pic_list']:
        image = cv2.imread(os.path.join(imageFolder, pic + '.jpg'))
        print(index, pic, image.shape[1], image.shape[0])
        imageResolutions[pic] = [image.shape[1], image.shape[0]]
print("Get Image Resolution Collections Done.")

with open(os.path.join("/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test/", "testResolution.json"), "w") as f:
    json.dump(imageResolutions, f)