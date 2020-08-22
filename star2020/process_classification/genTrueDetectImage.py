# -*- coding: utf-8 -*-
"""
@author: Jingxiao Gu
"""
import os
import cv2
import json



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


label1 = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112']
label2 = ['201', '202', '203', '204', '205', '206', '207']
label3 = ['301']

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

def parseJson(gtPath, pdPath, picPath, savePath, round):
    fJson = open(gtPath, encoding="utf-8")
    gtJson = json.load(fJson)
    fJson = open(pdPath, encoding="utf-8")
    pdJson = json.load(fJson)

    gtSigns = gtJson['signs']
    pdSigns = pdJson['signs']

    # Analysis
    count = 100 * round
    for pdSign in pdSigns:
        pdpic_id0 = pdSign['pic_id']
        pdtype0 = labels[pdSign['type']]
        pdBox0 = [pdSign['x'], pdSign['y'], pdSign['x'] + pdSign['w'], pdSign['y'] + pdSign['h']]
        for gtSign in gtSigns:
            pic_id0 = gtSign['pic_id']
            type0 = labels[gtSign['type']]
            gtBox0 = [gtSign['x'], gtSign['y'], gtSign['x'] + gtSign['w'], gtSign['y'] + gtSign['h']]

            if cal_iou(gtBox0, pdBox0) >= 0.5 and type0 == pdtype0 and pic_id0 == pdpic_id0:

                if pdtype0 == labels['102']:
                    count = count + 1
                    image = cv2.imread(os.path.join(picPath, pdpic_id0 + '.jpg'))

                    if pdSign['w'] > 20:
                        xmin = max(0, pdSign['x'] - int(pdSign['w'] * 0.05))
                        xmax = min(image.shape[1], pdSign['x'] + pdSign['w'] + int(pdSign['w'] * 0.05))
                    else:
                        xmin = max(0, pdSign['x'] - 1)
                        xmax = min(image.shape[1], pdSign['x'] + pdSign['w'] + 1)

                    if pdSign['h'] > 20:
                        ymin = max(0, pdSign['y'] - int(pdSign['h'] * 0.05))
                        ymax = min(image.shape[1], pdSign['y'] + pdSign['h'] + int(pdSign['h'] * 0.05))
                    else:
                        ymin = max(0, pdSign['y'] - 1)
                        ymax = min(image.shape[0], pdSign['y'] + pdSign['h'] + 1)

                    roi_image = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
                    roi_image = cv2.resize(roi_image, (160, 160))

                    cv2.imwrite(os.path.join(savePath, 'true', pdpic_id0 + '_' + str(count) + '_' + pdSign['type'] + '.png'), roi_image)



# 用于评估预测文件与标注文件的F1 Score
if __name__ == '__main__':
    # gtFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/val_tag/"
    # pdFolder = "../../output_results/val/"
    # picPath = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/pic/'
    # savePath = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/false_classification/207/val'
    # filterDets = ['ensemble_val_050']

    gtFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/train_tag/"
    pdFolder = "../../output_results/train/"
    picPath = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/pic/'
    savePath = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/false_classification/102/train'
    filterDets = ['ensemble_train_040']

    for i, filterDet in enumerate(filterDets):
        filterFolder = os.path.join(pdFolder, filterDet)
        for index, tag in enumerate(os.listdir(filterFolder)):
            print(index)
            gtPath = os.path.join(gtFolder, tag)
            pdPath = os.path.join(filterFolder, tag)
            parseJson(gtPath, pdPath, picPath, savePath, i)
