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


def idgettype(signid, signs_list):
    for sign_info in signs_list:
        if sign_info['sign_id'] == signid:
            return sign_info['type']

def deleteDuplicatedElementFromList(list):
    resultList = []
    for item in list:
        if len(resultList) == 0:
            resultList.append(item)
        else:
            flag = 1
            for item1 in resultList:
                if item == item1:
                    flag = 0
                else:
                    continue
            if flag == 1:
                resultList.append(item)
    return resultList


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


def parseJson(gtPath, pdPath):
    fJson = open(gtPath)
    gtJson = json.load(fJson)
    fJson = open(pdPath)
    pdJson = json.load(fJson)

    gtSigns = gtJson['signs']
    gtMatchs = gtJson['match']
    pdSigns = pdJson['signs']
    pdMatchs = pdJson['match']

    # No longer use ----------------------------------------------------
    # pdMatchsNew = []

    # # Delete Wrong Order
    # for pdMatch in pdMatchs:
    #     pdSign_id = pdMatch['sign_id']
    #     pdMatch_sign_id = pdMatch['match_sign_id']
    #     pdMatch_score = pdMatch['match_score']
    #
    #     if 'B' in pdSign_id and pdMatch_sign_id == " ":
    #         n = 0
    #         # print(pdSign_id, pdMatch_sign_id)
    #     elif 'B' in pdSign_id:
    #         pdMatchsNew.append({'sign_id': pdMatch_sign_id, 'match_sign_id': pdSign_id, 'match_score':pdMatch_score })
    #         # pdMatchsNew.append({'sign_id': pdMatch_sign_id, 'match_sign_id': pdSign_id})
    #     else:
    #         pdMatchsNew.append(pdMatch)
    # pdMatchs = pdMatchsNew
    #
    # # Delete Wrong Type
    # filter_match = []
    # for match_info in pdMatchs:
    #     sign_a = match_info['sign_id']
    #     sign_b = match_info['match_sign_id']
    #     # 根据id确度type
    #     sign_a_type = idgettype(sign_a, pdSigns)
    #     sign_b_type = idgettype(sign_b, pdSigns)
    #     sign_score = match_info['match_score']
    #
    #     new_match = dict()
    #     if sign_a_type == sign_b_type:
    #         new_match['sign_id'] = sign_a
    #         new_match['match_sign_id'] = sign_b
    #         new_match['match_score'] = sign_score
    #         filter_match.append(new_match)
    #
    # for match_info in pdMatchs:
    #     sign_a = match_info['sign_id']
    #     sign_b = match_info['match_sign_id']
    #     # 根据id确度type
    #     sign_a_type = idgettype(sign_a, pdSigns)
    #     sign_b_type = idgettype(sign_b, pdSigns)
    #     sign_score = match_info['match_score']
    #
    #     new_match = dict()
    #     if sign_a_type != sign_b_type:
    #         ids = []
    #         for match in filter_match:
    #             ids.append(match['sign_id'])
    #         if sign_a not in ids:
    #             new_match['sign_id'] = sign_a
    #             new_match['match_sign_id'] = " "
    #             new_match['match_score'] = sign_score
    #             filter_match.append(new_match)
    #
    # pdMatchs = filter_match
    # pdMatchs = deleteDuplicatedElementFromList(pdMatchs)
    #---------------------------------------------------------------------

    thresPdSigns = []
    thresPdMatchs = []
    sign_id_list = []
    for pdSign in pdSigns:
        if 'score' not in pdSign:
            thresPdSigns.append(pdSign)
            sign_id_list.append(pdSign['sign_id'])
        else:
            if pdSign['score'] >= threshold:
                thresPdSigns.append(pdSign)
                sign_id_list.append(pdSign['sign_id'])
    pdSigns = thresPdSigns
    sign_id_list.append(" ")

    for pdMatch in pdMatchs:
        if pdMatch['sign_id'] in sign_id_list and pdMatch['match_sign_id'] in sign_id_list:
            thresPdMatchs.append(pdMatch)
    pdMatchs = thresPdMatchs

    tp = 0
    total_tp = len(gtMatchs)
    total_pd = len(pdMatchs)

    tpCounts = [0] * len(gtMatchs)

    tpClassCounts = [0] * 19
    gtClassCounts = [0] * 19
    pdClassCounts = [0] * 19

    for pdMatch in pdMatchs:
        sign_id0 = pdMatch['sign_id']
        for pdSign in pdSigns:
            if pdSign['sign_id'] == sign_id0:
                pdtype0 = pdSign['type']

        pdClassCounts[labels[pdtype0]] = pdClassCounts[labels[pdtype0]] + 1

    for tp_idx, gtMatch in enumerate(gtMatchs):
        sign_id = gtMatch['sign_id']
        match_sign_id = gtMatch['match_sign_id']

        gtBox0 = [0, 0, 10000, 10000]
        gtBox1 = [0, 0, 10000, 10000]
        pic_id0 = ''
        pic_id1 = ''
        type0 = 0
        type1 = 1
        for gtSign in gtSigns:
            pic_id = gtSign['pic_id']
            if gtSign['sign_id'] == sign_id:
                pic_id0 = pic_id
                type0 = gtSign['type']
                gtBox0 = [gtSign['x'], gtSign['y'], (gtSign['x'] + gtSign['w']), (gtSign['y'] + gtSign['h'])]

            if gtSign['sign_id'] == match_sign_id:
                pic_id1 = pic_id
                type1 = gtSign['type']
                gtBox1 = [gtSign['x'], gtSign['y'], (gtSign['x'] + gtSign['w']), (gtSign['y'] + gtSign['h'])]

        gtClassCounts[labels[type0]] = gtClassCounts[labels[type0]] + 1

        for pdMatch in pdMatchs:
            sign_id0 = pdMatch['sign_id']
            match_sign_id0 = pdMatch['match_sign_id']

            pdBox0 = [0, 0, 1, 1]
            pdBox1 = [0, 0, 1, 1]
            pdpic_id0 = ''
            pdpic_id1 = ''
            pdtype0 = 5000
            pdtype1 = 6000

            for pdSign in pdSigns:
                pdpic_id = pdSign['pic_id']
                if pdSign['sign_id'] == sign_id0:
                    pdpic_id0 = pdpic_id
                    pdtype0 = pdSign['type']
                    pdBox0 = [pdSign['x'], pdSign['y'], (pdSign['x'] + pdSign['w']), (pdSign['y'] + pdSign['h'])]
                if pdSign['sign_id'] == match_sign_id0:
                    pdpic_id1 = pdpic_id
                    pdtype1 = pdSign['type']
                    pdBox1 = [pdSign['x'], pdSign['y'], (pdSign['x'] + pdSign['w']), (pdSign['y'] + pdSign['h'])]

            if cal_iou(gtBox0, pdBox0) >= 0.5 and type0 == pdtype0 and pic_id0 == pdpic_id0 and \
                cal_iou(gtBox1, pdBox1) >= 0.5 and type1 == pdtype1 and pic_id1 == pdpic_id1:
                tpCounts[tp_idx] = 1
                tpClassCounts[labels[pdtype0]] = tpClassCounts[labels[pdtype0]] + 1
            elif cal_iou(gtBox0, pdBox0) >= 0.5 and type0 == pdtype0 and pic_id0 == pdpic_id0 and \
                    match_sign_id == ' ' and  match_sign_id0 == ' ':
                tpCounts[tp_idx] = 1
                tpClassCounts[labels[pdtype0]] = tpClassCounts[labels[pdtype0]] + 1
            # elif cal_iou(gtBox1, pdBox1) >= 0.5 and type1 == pdtype1 and pic_id1 == pdpic_id1 and \
            #         sign_id == ' ' and  sign_id0 == ' ':
            #     print(1)
            #     tpCounts[tp_idx] = 1

    for tpc in tpCounts:
        tp = tp + tpc

    fn = total_tp - tp
    fp = total_pd - tp
    # print(total_tp, total_pd, tp, fn, fp)

    if tp + fp == 0:
        P = 0
    else:
        P = float(tp) / (tp + fp)

    if tp + fn == 0:
        R = 0
    else:
        R = float(tp) / (tp + fn)

    if P + R == 0:
        f1 = 0
    else:
        f1 = float(2 * P * R) / (P + R)
    return f1, P, R, total_tp, total_pd, tp, tpClassCounts, gtClassCounts, pdClassCounts

# 用于评估预测文件与标注文件的F1 Score
if __name__ == '__main__':
    gtFolder = "../../output_results/tag/val_tag/"
    pdFolder = "../../output_results/val/ensemble_val_050_match_check/"

    thresholds = []
    f1_plot = []
    p_plot = []
    r_plot = []
    for i in range(50, 85, 5):
        thresholds.append(i / 100.0)

    for threshold in thresholds:
        F1_group = 0
        P_group = 0
        R_group = 0
        n_group = 0

        gt_general = 0
        pd_general = 0
        tp_general = 0
        F1_general = 0
        P_general = 0
        R_general = 0

        TpClass = np.array([0.0] * 19)
        GtClass = np.array([0.0] * 19)
        PdClass = np.array([0.0] * 19)
        classF1 = np.array([0.0] * 19)
        P_class = np.array([0.0] * 19)
        R_class = np.array([0.0] * 19)
        F1_class = np.array([0.0] * 19)

        for index, tag in enumerate(os.listdir(gtFolder)):
            n_group = n_group + 1
            gtPath = os.path.join(gtFolder, tag)
            pdPath = os.path.join(pdFolder, tag)
            f1, p, r, total_gt, total_pd, total_tp, tpClassCounts, gtClassCounts, pdClassCounts = parseJson(gtPath, pdPath)
            F1_group = F1_group + f1
            P_group = P_group + p
            R_group = R_group + r

            gt_general = gt_general + total_gt
            pd_general = pd_general + total_pd
            tp_general = tp_general + total_tp

            TpClass = TpClass + np.array(tpClassCounts)
            GtClass = GtClass + np.array(gtClassCounts)
            PdClass = PdClass + np.array(pdClassCounts)

        # print(TpClassCounts, GtClassCounts, PdClassCounts)

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

        # print(TpClass, GtClass, PdClass)
        # print(F1_class_average, P_class_average, R_class_average)
        # General Average
        fn = gt_general - tp_general
        fp = pd_general - tp_general
        if tp_general + fp == 0:
            P_general = 0
        else:
            P_general = tp_general / (tp_general + fp)

        if tp_general + fn == 0:
            R_general = 0
        else:
            R_general = tp_general / (tp_general + fn)

        if P_general + R_general == 0:
            F1_general = 0
        else:
            F1_general = (2 * P_general * R_general) / (P_general + R_general)

        f1_plot.append(F1_group / n_group)
        p_plot.append(P_group / n_group)
        r_plot.append(R_group / n_group)

        print("-" * 158)
        print("Group Average whith threshold {:.4f}   - F1: {:.4f}, P: {:.4f}, R: {:.4f}".format(
            threshold, F1_group / n_group, P_group / n_group, R_group / n_group))
        print("General Average whith threshold {:.4f} - F1: {:.4f}, P: {:.4f}, R: {:.4f}".format(
            threshold, F1_general, P_general, R_general))
        print("Class Average whith threshold {:.4f}   - F1: {:.4f}, P: {:.4f}, R: {:.4f}".format(
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