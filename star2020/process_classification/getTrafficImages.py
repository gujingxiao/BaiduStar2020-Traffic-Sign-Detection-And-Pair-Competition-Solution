# -*- coding: utf-8 -*-
"""
Created on Tues Jun 30 10:20:01 2020

@author: Jingxiao Gu
"""

import cv2
import json
import numpy as np
import os
import random

def cal_iou(box1, box2):
    """
    :param box1: = [xmin1, ymin1, xmax1, ymax1]
    :param box2: = [xmin2, ymin2, xmax2, ymax2]
    :return:
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h  # C∩G的面积
    iou = area / (s1 + s2 - area)
    return iou

def parseJson(tagPath, tag, picFolder, saveFolder, width, height):
    fJson = open(tagPath, encoding="utf-8")
    tagJson = json.load(fJson)
    for index, seq0_pic in enumerate(tagJson['group'][0]['pic_list']):
        image = cv2.imread(os.path.join(picFolder, seq0_pic + '.jpg'))

        for sign in tagJson['signs']:
            if sign['pic_id'] == seq0_pic:
                if sign['w'] > 20:
                    xmin = int(max(0, sign['x'] - int(sign['w'] * 0.05)))
                    xmax = int(min(image.shape[1], sign['x'] + sign['w'] + int(sign['w'] * 0.05)))
                else:
                    xmin = int(max(0, sign['x'] - 1))
                    xmax = int(min(image.shape[1], sign['x'] + sign['w'] + 1))

                if sign['h'] > 20:
                    ymin = int(max(0, sign['y'] - int(sign['h'] * 0.05)))
                    ymax = int(min(image.shape[1], sign['y'] + sign['h'] + int(sign['h'] * 0.05)))
                else:
                    ymin = int(max(0, sign['y'] - 1))
                    ymax = int(min(image.shape[0], sign['y'] + sign['h'] + 1))

                # if sign['w'] > 20:
                #     xmin = int(max(0, sign['x'] - int(sign['w'] * 0.5)))
                #     xmax = int(min(image.shape[1], sign['x'] + sign['w'] + int(sign['w'] * 0.5)))
                # else:
                #     xmin = int(max(0, sign['x'] - 8))
                #     xmax = int(min(image.shape[1], sign['x'] + sign['w'] + 8))
                #
                # if sign['h'] > 20:
                #     ymin = int(max(0, sign['y'] - int(sign['h'] * 0.5)))
                #     ymax = int(min(image.shape[1], sign['y'] + sign['h'] + int(sign['h'] * 0.5)))
                # else:
                #     ymin = int(max(0, sign['y'] - 8))
                #     ymax = int(min(image.shape[0], sign['y'] + sign['h'] + 8))

                roi_image = image[ymin:ymax, xmin:xmax, :]
                roi_image = cv2.resize(roi_image, (width, height))

                label = sign['type']
                if os.path.exists(os.path.join(saveFolder, label)):
                    cv2.imwrite(os.path.join(saveFolder, label, tag.split('.')[0] + '_' + seq0_pic + '_' + sign['sign_id'] + '_' + sign['type'] + '.png'), roi_image)
                else:
                    os.mkdir(os.path.join(saveFolder, label))
                    cv2.imwrite(os.path.join(saveFolder, label, tag.split('.')[0] + '_' + seq0_pic + '_' + sign['sign_id'] + '_' + sign['type'] + '.png'),roi_image)

    for index, seq1_pic in enumerate(tagJson['group'][1]['pic_list']):
        image = cv2.imread(os.path.join(picFolder, seq1_pic + '.jpg'))
        gtBoxes = []
        for sign in tagJson['signs']:
            if sign['pic_id'] == seq1_pic:
                if sign['w'] > 20:
                    xmin = int(max(0, sign['x'] - int(sign['w'] * 0.05)))
                    xmax = int(min(image.shape[1], sign['x'] + sign['w'] + int(sign['w'] * 0.05)))
                else:
                    xmin = int(max(0, sign['x'] - 1))
                    xmax = int(min(image.shape[1], sign['x'] + sign['w'] + 1))

                if sign['h'] > 20:
                    ymin = int(max(0, sign['y'] - int(sign['h'] * 0.05)))
                    ymax = int(min(image.shape[1], sign['y'] + sign['h'] + int(sign['h'] * 0.05)))
                else:
                    ymin = int(max(0, sign['y'] - 1))
                    ymax = int(min(image.shape[0], sign['y'] + sign['h'] + 1))

                # if sign['w'] > 20:
                #     xmin = int(max(0, sign['x'] - int(sign['w'] * 0.5)))
                #     xmax = int(min(image.shape[1], sign['x'] + sign['w'] + int(sign['w'] * 0.5)))
                # else:
                #     xmin = int(max(0, sign['x'] - 8))
                #     xmax = int(min(image.shape[1], sign['x'] + sign['w'] + 8))
                #
                # if sign['h'] > 20:
                #     ymin = int(max(0, sign['y'] - int(sign['h'] * 0.5)))
                #     ymax = int(min(image.shape[1], sign['y'] + sign['h'] + int(sign['h'] * 0.5)))
                # else:
                #     ymin = int(max(0, sign['y'] - 8))
                #     ymax = int(min(image.shape[0], sign['y'] + sign['h'] + 8))

                gtBoxes.append([xmin, ymin, xmax, ymax])

                roi_image = image[ymin:ymax, xmin:xmax, :]
                roi_image = cv2.resize(roi_image, (width, height))

                label = sign['type']
                if os.path.exists(os.path.join(saveFolder, label)):
                    cv2.imwrite(os.path.join(saveFolder, label, tag.split('.')[0] + '_' + seq1_pic + '_' + sign['sign_id'] + '_' + sign['type'] + '.png'), roi_image)
                else:
                    os.mkdir(os.path.join(saveFolder, label))
                    cv2.imwrite(os.path.join(saveFolder, label, tag.split('.')[0] + '_' + seq1_pic + '_' + sign['sign_id'] + '_' + sign['type'] + '.png'), roi_image)


if __name__ == '__main__':
    tagFolder = "ensemble_val_050_match"
    picFolder = "pic"
    baseFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/"
    saveFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/final_classification/val/"
    width, height = 160, 160

    for index, tag in enumerate(os.listdir(os.path.join(baseFolder, tagFolder))):
        tagPath = os.path.join(baseFolder, tagFolder, tag)
        parseJson(tagPath, tag, os.path.join(baseFolder, picFolder), saveFolder, width, height)
        if index % 100 == 0:
            print(index, '/', len(os.listdir(os.path.join(baseFolder, tagFolder))))
