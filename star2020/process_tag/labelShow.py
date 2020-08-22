# -*- coding: utf-8 -*-
"""
Created on Tues Jun 30 10:20:01 2020

@author: Jingxiao Gu
"""

import cv2
import json
import numpy as np
import os

COLORS = ((244,  67,  54),
          (255, 193,   7),
          (  0, 150, 136),
          (158, 158, 158),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          ( 96, 125, 139))

fontScale = 1
fontcolor = (255, 0, 0)  # BGR
thickness = 2
lineType = 4

def parseJson(tagPath, picFolder, showWidth, showHeight):
    fJson = open(tagPath)
    tagJson = json.load(fJson)
    postSigns = tagJson['signs']

    lenSeq0 = len(tagJson['group'][0]['pic_list'])
    lenSeq1 = len(tagJson['group'][1]['pic_list'])

    maxLen = max(lenSeq0, lenSeq1)
    scaleHeight = int(showHeight / maxLen)

    backGround = np.zeros((showHeight, showWidth, 3), dtype=np.uint8)
    # print(tagPath)
    # print(tagJson['group'][0]['pic_list'])
    # print(tagJson['group'][1]['pic_list'])
    for index, seq0_pic in enumerate(tagJson['group'][0]['pic_list']):

        if os.path.exists(os.path.join(picFolder, seq0_pic + '.jpg')):
            image = cv2.imread(os.path.join(picFolder, seq0_pic + '.jpg'))
        else:
            # print(seq0_pic)
            image = cv2.imread(os.path.join(picFolder, seq0_pic + '.png'))

        cv2.putText(image, seq0_pic, (10, 30), cv2.FONT_HERSHEY_COMPLEX, fontScale, fontcolor, thickness, lineType)

        wScale = showWidth / 2 / image.shape[1]
        hScale = scaleHeight / image.shape[0]
        for sign in postSigns:
            if sign['pic_id'] == seq0_pic:
                sign['h'] = int(sign['h'] * hScale)
                sign['w'] = int(sign['w'] * wScale)
                sign['x'] = int(sign['x'] * wScale) + 0
                sign['y'] = int(sign['y'] * hScale) + index * scaleHeight

        scaleImage = cv2.resize(image, (int(showWidth / 2), scaleHeight))

        backGround[index * scaleHeight: (index + 1) * scaleHeight, 0: int(showWidth / 2), :] = scaleImage

    for index, seq1_pic in enumerate(tagJson['group'][1]['pic_list']):

        if os.path.exists(os.path.join(picFolder, seq1_pic + '.jpg')):
            image = cv2.imread(os.path.join(picFolder, seq1_pic + '.jpg'))
        else:
            # print(seq1_pic)
            image = cv2.imread(os.path.join(picFolder, seq1_pic + '.png'))

        wScale = showWidth / 2 / image.shape[1]
        hScale = scaleHeight / image.shape[0]
        for sign in postSigns:
            if sign['pic_id'] == seq1_pic:
                sign['h'] = int(sign['h'] * hScale)
                sign['w'] = int(sign['w'] * wScale)
                sign['x'] = int(sign['x'] * wScale) + int(showWidth / 2)
                sign['y'] = int(sign['y'] * hScale) + index * scaleHeight

        cv2.putText(image, seq1_pic, (10, 30), cv2.FONT_HERSHEY_COMPLEX, fontScale, fontcolor, thickness, lineType)
        scaleImage = cv2.resize(image, (int(showWidth / 2), scaleHeight))

        backGround[index * scaleHeight: (index + 1) * scaleHeight, int(showWidth / 2): , :] = scaleImage

    for sign in postSigns:
        cv2.rectangle(backGround, (sign['x'], sign['y']),  (sign['x'] + sign['w'], sign['y'] + sign['h']), (0, 255, 0), 2)

    for index, sign in enumerate(postSigns):
        while index >= len(COLORS):
            index = index - len(COLORS)
            # print(index)
        sign['color'] =COLORS[index]

    for match in tagJson['match']:
        sign_id = match['sign_id']
        match_sign_id = match['match_sign_id']

        rect0 = []
        rect1 = []

        if match_sign_id != '' and match_sign_id != ' ':
            color = (255, 255, 255)
            sign0 = ''
            sign1 = ''
            score = ''
            for sign in postSigns:
                if sign['sign_id'] == sign_id:
                    rect0 = [sign['x'], sign['y'] ,sign['x'] + sign['w'], sign['y'] + sign['h']]
                    color = sign['color']
                    sign0 = sign['type']

                if sign['sign_id'] == match_sign_id:
                    rect1 = [sign['x'], sign['y'] ,sign['x'] + sign['w'], sign['y'] + sign['h']]
                    sign1 = sign['type']
            score = str(match['match_score'])
            cv2.putText(backGround, sign0, (rect0[0], rect0[1] - 10), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness, lineType)
            cv2.putText(backGround, score, ((rect0[0] + rect1[0]) // 2, (rect0[1] + rect1[1])//2 ), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1, lineType)
            cv2.putText(backGround, sign1, (rect1[0], rect1[1] - 10), cv2.FONT_HERSHEY_COMPLEX, fontScale, color, thickness, lineType)
            cv2.line(backGround, (rect0[2], rect0[3]), (rect1[0], rect1[1]), color, 2)

    return backGround



if __name__ == '__main__':
    tagFolder = "/home/gujingxiao/projects/Paddle Solution Backup/PaddleDetection/det_results/VAL/ensemble_val_match_check/"
    picFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/pic"
    inputFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/input/"
    saveFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/pair_val_images/"
    showWidth, showHeight = 2000, 5000

    for tag in os.listdir(tagFolder):
        tagPath = os.path.join(tagFolder, tag)
        labelShowImage = parseJson(tagPath, picFolder, showWidth, showHeight)
        cv2.imwrite(os.path.join(saveFolder, tag.replace('.json', '.jpg')), labelShowImage)