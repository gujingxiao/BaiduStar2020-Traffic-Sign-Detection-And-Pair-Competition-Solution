# -*- coding: utf-8 -*-
"""
@author: Jingxiao Gu
"""
import os
import numpy as np
import json
import matplotlib.pyplot as plt

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

def parseJson(gtPath, pdPath, threshold):
    fJson = open(gtPath, encoding="utf-8")
    gtJson = json.load(fJson)
    fJson = open(pdPath, encoding="utf-8")
    pdJson = json.load(fJson)

    gtSigns = gtJson['signs']
    pdSigns = pdJson['signs']

    thresPdSigns = []
    tpClassCounts = [0] * 19
    gtClassCounts = [0] * 19
    pdClassCounts = [0] * 19

    for pdSign in pdSigns:
        # if pdSign['cls_score'] * 0.3 + pdSign['score'] * 0.7 >= threshold:
        if pdSign['score'] >= threshold:
            thresPdSigns.append(pdSign)
            pdtype0 = pdSign['type']
            pdClassCounts[labels[pdtype0]] = pdClassCounts[labels[pdtype0]] + 1
    pdSigns = thresPdSigns

    tp = 0
    total_tp = len(gtSigns)
    total_pd = len(pdSigns)

    for gtSign in gtSigns:
        pic_id0 = gtSign['pic_id']

        type0 = gtSign['type']
        gtBox0 = [gtSign['x'], gtSign['y'], gtSign['x'] + gtSign['w'], gtSign['y'] + gtSign['h']]
        gtClassCounts[labels[type0]] = gtClassCounts[labels[type0]] + 1

        for pdSign in pdSigns:
            pdpic_id0 = pdSign['pic_id']
            pdtype0 = pdSign['type']
            pdBox0 = [pdSign['x'], pdSign['y'], pdSign['x'] + pdSign['w'], pdSign['y'] + pdSign['h']]

            if cal_iou(gtBox0, pdBox0) >= 0.5 and type0 == pdtype0 and pic_id0 == pdpic_id0:
                tp += 1
                tpClassCounts[labels[pdtype0]] = tpClassCounts[labels[pdtype0]] + 1
                break

    fn = total_tp - tp
    fp = total_pd - tp
    # print(total_tp, total_pd, tp, fn, fp)

    if tp + fp == 0:
        P = 0
    else:
        P = tp / (tp + fp)

    if tp + fn == 0:
        R = 0
    else:
        R = tp / (tp + fn)

    if P + R == 0:
        f1 = 0
    else:
        f1 = (2 * P * R) / (P + R)
    return f1, P, R, tpClassCounts, gtClassCounts, pdClassCounts

# 用于评估预测文件与标注文件的F1 Score
if __name__ == '__main__':
    gtFolder = "../../output_results/tag/val_tag/"
    pdFolder = "../../output_results/val/ensemble_val_050_filter"

    thresholds = []
    f1_plot = []
    p_plot = []
    r_plot = []
    for i in range(50, 95, 5):
        thresholds.append(i / 100.0)

    for threshold in thresholds:
        F1 = 0
        P = 0
        R = 0
        n = 0

        TpClass = np.array([0.0] * 19)
        GtClass = np.array([0.0] * 19)
        PdClass = np.array([0.0] * 19)
        classF1 = np.array([0.0] * 19)
        P_class = np.array([0.0] * 19)
        R_class = np.array([0.0] * 19)
        F1_class = np.array([0.0] * 19)

        for index, tag in enumerate(os.listdir(pdFolder)):
            n = n + 1
            gtPath = os.path.join(gtFolder, tag)
            pdPath = os.path.join(pdFolder, tag)
            f1, p, r, tpClassCounts, gtClassCounts, pdClassCounts = parseJson(gtPath, pdPath, threshold)
            F1 = F1 + f1
            P = P + p
            R = R + r

            TpClass = TpClass + np.array(tpClassCounts)
            GtClass = GtClass + np.array(gtClassCounts)
            PdClass = PdClass + np.array(pdClassCounts)

        # Class Average
        fn_class = GtClass - TpClass
        fp_class = PdClass - TpClass

        for index in range(19):
            if TpClass[index] + fp_class[index] == 0:
                P_class[index] = 0
            else:
                P_class[index] = TpClass[index] / PdClass[index] #(TpClass[index] + fp_class[index])

            if TpClass[index] + fn_class[index] == 0:
                R_class[index] = 0
            else:
                R_class[index] = TpClass[index] / (TpClass[index] + fn_class[index])

            if P_class[index] + R_class[index] == 0:
                F1_class[index] = 0
            else:
                F1_class[index] = (2 * P_class[index] * R_class[index]) / (P_class[index] + R_class[index])
        valid_class = 0
        for value in GtClass:
            if value > 0:
                valid_class += 1
        F1_class_average = np.sum(F1_class) / valid_class
        P_class_average = np.sum(P_class) / valid_class
        R_class_average = np.sum(R_class) / valid_class

        f1_plot.append(F1 / n)
        p_plot.append(P / n)
        r_plot.append(R / n)

        print("Threshold {:.4f} - F1: {:.4f}, P: {:.4f}, R: {:.4f}".format(threshold, F1 / n, P / n, R / n))
        print("Class Average whith threshold {:.4f}   - F1: {:.4f}, P: {:.4f}, R: {:.4f},".format(
            threshold, F1_class_average, P_class_average, R_class_average))
        print("| No.|   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   10  |   11  |   12  |   13  |   14  |   15  |   16  |   17  |   18  |")
        print("| F1 |{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|".format(
            F1_class[0],F1_class[1],F1_class[2],F1_class[3],F1_class[4],F1_class[5],F1_class[6],F1_class[7],F1_class[8],F1_class[9],F1_class[10],F1_class[11]
            , F1_class[12],F1_class[13],F1_class[14],F1_class[15],F1_class[16],F1_class[17],F1_class[18]))
        print("| P  |{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|".format(
            P_class[0],P_class[1],P_class[2],P_class[3],P_class[4],P_class[5],P_class[6],P_class[7],P_class[8],P_class[9],P_class[10],P_class[11]
            , P_class[12],P_class[13],P_class[14],P_class[15],P_class[16],P_class[17],P_class[18]))
        print("| R  |{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|{:.5f}|".format(
            R_class[0],R_class[1],R_class[2],R_class[3],R_class[4],R_class[5],R_class[6],R_class[7],R_class[8],R_class[9],R_class[10],R_class[11]
            , R_class[12],R_class[13],R_class[14],R_class[15],R_class[16],R_class[17],R_class[18]))
        print("-" * 158 + '\n')

    thresholds = np.array(thresholds)
    f1_plot = np.array(f1_plot)
    p_plot = np.array(p_plot)
    r_plot = np.array(r_plot)

    plt.plot(thresholds, f1_plot, label='f1', marker='o', c='r')
    plt.plot(thresholds, p_plot, label='p', marker='x', c='g')
    plt.plot(thresholds, r_plot, label='r', marker='*', c='b')
    plt.legend()
    plt.show()

