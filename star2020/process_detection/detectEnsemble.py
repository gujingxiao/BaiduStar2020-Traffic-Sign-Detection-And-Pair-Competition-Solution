import os
import numpy as np
import pandas as pd

import os
import json
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import glob
import pickle

"""
Ensembling methods for object detection.
"""

""" 
General Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences. Can weigh different detectors with different weights.
No real learning here, although the weights and iou_thresh can be optimized.

Input: 
 - dets : List of detections. Each detection is all the output from one detector, and
          should be a list of boxes, where each box should be on the format 
          [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y 
          are the center coordinates, box_w and box_h are width and height resp.
          The values should be floats, except the class which should be an integer.

 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same, 
               if they also belong to the same class.

 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. The list should be as long as the
            number of detections. If this is set to None, then all detectors
            will be considered equally reliable. The sum of weights does not
            necessarily have to be 1.

Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""


def py_cpu_nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def GeneralEnsemble(dets, iou_thresh=0.55, weights=None):
    assert (type(iou_thresh) == float)

    ndets = len(dets)

    if weights is None:
        w = 1 / float(ndets)
        weights = [w] * ndets
    else:
        assert (len(weights) == ndets)

        s = sum(weights)
        for i in range(0, len(weights)):
            weights[i] /= s

    out = list()
    used = list()

    for idet in range(0, ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue

            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]

                if odet == det:
                    continue

                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        if box[4] == obox[4]:
                            # Same class
                            iou = computeIOU(box, obox)
                            if iou > bestiou:
                                bestiou = iou
                                bestbox = obox

                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox, w))
                    used.append(bestbox)

            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                new_box[5] /= ndets
                out.append(new_box)
            else:
                allboxes = [(box, weights[idet])]
                allboxes.extend(found)

                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0

                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    wsum += w

                    b = bb[0]
                    xc += w * b[0]
                    yc += w * b[1]
                    bw += w * b[2]
                    bh += w * b[3]
                    conf += w * b[5]

                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                new_box = [xc, yc, bw, bh, box[4], conf]
                out.append(new_box)
    return out


def getCoords(box):
    x1 = float(box[0]) - float(box[2]) / 2
    x2 = float(box[0]) + float(box[2]) / 2
    y1 = float(box[1]) - float(box[3]) / 2
    y2 = float(box[1]) + float(box[3]) / 2
    return x1, x2, y1, y2


def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1)
    x21, x22, y21, y22 = getCoords(box2)

    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    iou = intersect_area / (box1_area + box2_area - intersect_area + 1e-6)
    return iou


def get_dets_by_pred_str(pred_str):
    try:
        if len(pred_str) < 1:
            return []
    except:
        print('pred_str:', pred_str)
        raise
    det_items = pred_str.split(' ')
    dets, det = [], []
    for i in range(len(det_items)):
        det.append(det_items[i])
        if (i+1) % 6 == 0:
            label = det[0]
            conf = float(det[1])
            xmin = float(det[2])
            ymin = float(det[3])
            xmax = float(det[4])
            ymax = float(det[5])
            new_det = [(xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin, label, conf]
            dets.append(new_det)
            det = []
    return dets

def xywh_to_pre_str(dets):
    res = []
    if len(dets) < 1:
        return ''
    for det in dets:
        label = det[4]
        conf = det[5]
        xmin = det[0] - (det[2] / 2)
        xmax = det[0] + (det[2] / 2)
        ymin = det[1] - (det[3] / 2)
        ymax = det[1] + (det[3] / 2)

        res.append(label)
        res.append('{:.4f}'.format(conf))
        res.append('{:.4f}'.format(xmin))
        res.append('{:.4f}'.format(ymin))
        res.append('{:.4f}'.format(xmax))
        res.append('{:.4f}'.format(ymax))

    res = [str(x) for x in res]
    return ' '.join(res)


if __name__ == "__main__":
    # Train
    # saveFolder = "../../output_results/train/ensemble_train"
    # baseFolder = "../../output_results/train/"
    # detNames = ['cascade_cbr50_c3c19_class_new', 'cascade_r2n200_c3c19_class_new',
    #             'cascade_r200_c3c19_class_new']

    # Val
    saveFolder = "../../output_results/val/ensemble_val2"
    baseFolder = "../../output_results/val/"
    detNames = ['cascade_cbr50_c3c19_val', 'cascade_cbr200_c3c19_val', 'cascade_r2n200_c3c19_val',
                 'cascade_r200_c3c19_val', 'ensemble_val1', 'ensemble_val2']  # 1.0
    detNames = ['cascade_r152_c3c19_val', 'cascade_x101_c3c19_val']  # 1.2
    detNames = ['cascade_r2n101_c3c19_val', 'cascade_senet154_c3c19_val']  # 1.1
    factor = 1.1

    # Test
    # saveFolder = "../../output_results/test/ensemble_test"
    # baseFolder = "../../output_results/test/"
    # detNames = ['cascade_cbr50_c3c19_test', 'cascade_cbr200_c3c19_test', 'cascade_r2n200_c3c19_test',
    #             'cascade_r200_c3c19_test', 'ensemble_test1', 'ensemble_test2']  # 1.0
    # detNames = ['cascade_r152_c3c19_test', 'cascade_x101_c3c19_test']  # 1.2
    # detNames = ['cascade_r2n101_c3c19_test', 'cascade_se154_c3c19_test']  # 1.1

    length = len(detNames)
    thres = [0.2] * length
    weights = [1.0 / length] * length
    for index, tag in enumerate(os.listdir(os.path.join(baseFolder, detNames[0]))):
        detJsons = []
        for detName in detNames:
            fJson = open(os.path.join(baseFolder, detName, tag), encoding="utf-8")
            detJson = json.load(fJson)
            detJsons.append(detJson)

        pic_list = []
        for pic_name in detJsons[0]['group'][0]['pic_list']:
            pic_list.append(pic_name)
        for pic_name in detJsons[0]['group'][1]['pic_list']:
            pic_list.append(pic_name)

        newSigns = []
        for pic_name in pic_list:
            total_dets = []
            for index in range(len(detNames)):
                total_dets.append([])


            for index, detJson in enumerate(detJsons):
                for s in detJson['signs']:
                    if s['pic_id'] == pic_name:
                        if s['score'] >= thres[index]:
                            total_dets[index].append([s['x'], s['y'], s['w'], s['h'], s['type'], min(0.99, s['score'] * factor)])

            if len(weights) != len(total_dets):
                print(weights, total_dets)
            ens_dets = GeneralEnsemble(total_dets, iou_thresh=0.5, weights=weights)

            np_ens_dets = []
            for ens in ens_dets:
                np_ens_dets.append([ens[0], ens[1], ens[2] + ens[0], ens[3] + ens[1], ens[5]])

            np_ens_dets = np.array(np_ens_dets)
            if len(np_ens_dets) > 0:
                keeps = py_cpu_nms(np_ens_dets, 0.3)
                for keep in keeps:
                    ens = ens_dets[keep]
                    newSigns.append({'x': ens[0], 'y': ens[1], 'w': ens[2], 'h': ens[3], 'type': ens[4], 'score': ens[5], 'pic_id': pic_name})

        detJsons[0]['signs'] = newSigns
        with open(os.path.join(saveFolder, tag), "w") as f:
            json.dump(detJsons[0], f)