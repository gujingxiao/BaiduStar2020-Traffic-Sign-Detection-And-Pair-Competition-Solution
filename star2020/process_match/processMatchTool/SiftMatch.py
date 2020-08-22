import numpy as np
import cv2
import math
import imagehash
from PIL import Image
from .hazeRemoval import deHaze


def matchCenterError(desPoints, matchesMask0, matches0, roi, gtbox, mode=0):
    gtbox = [gtbox[0] - roi[0], gtbox[1] - roi[1], gtbox[2] - roi[0], gtbox[3] - roi[1]]
    length = 0
    total_scores_w = 0
    total_scores_h = 0

    center_w = gtbox[0] / (roi[2] - roi[0])
    center_h = gtbox[1] / (roi[3] - roi[1])

    if mode == 0:
        for index, kp in enumerate(desPoints):
            w = kp.pt[0]
            h = kp.pt[1]
            if matchesMask0[index] == [1, 0]:
                length += 1
                if w >= gtbox[0] and w <= gtbox[2] and h >= gtbox[1] and h <= gtbox[3]:
                    w = w / (roi[2] - roi[0])
                    h = h / (roi[3] - roi[1])
                else:
                    w = w / (roi[2] - roi[0])
                    h = h / (roi[3] - roi[1])

                total_scores_w += w
                total_scores_h += h
    else:
        for match in matches0:
            gt_idx = match[1].trainIdx
            kp = desPoints[gt_idx]
            w = kp.pt[0]
            h = kp.pt[1]
            length += 1
            if w >= gtbox[0] and w <= gtbox[2] and h >= gtbox[1] and h <= gtbox[3]:
                w = w / (roi[2] - roi[0])
                h = h / (roi[3] - roi[1])
            else:
                w = w / (roi[2] - roi[0])
                h = h / (roi[3] - roi[1])

            total_scores_w += w
            total_scores_h += h

    if length == 0:
        error = 1.0
        return error
    else:
        total_scores_w = total_scores_w / length
        total_scores_h = total_scores_h / length

        error = math.sqrt(pow(total_scores_w - center_w, 2) + pow(total_scores_h - center_h, 2))
        return error


def getSiftMatchScore(image_sign, image_sign_match, gtBox, negBoxes, gtBox_match, offset_ratio):
    w_offset = int(offset_ratio * (gtBox[2] - gtBox[0]))
    h_offset = int(offset_ratio * (gtBox[3] - gtBox[1]))
    gtw_offset = int(offset_ratio * (gtBox_match[2] - gtBox_match[0]))
    gth_offset = int(offset_ratio * (gtBox_match[3] - gtBox_match[1]))
    wm_offset = int(0.1 *  (gtBox[2] - gtBox[0]))
    hm_offset = int(0.1 * (gtBox[3] - gtBox[1]))
    gtwm_offset = int(0.1 *  (gtBox_match[2] - gtBox_match[0]))
    gthm_offset = int(0.1 * (gtBox_match[3] - gtBox_match[1]))
    h0, w0 = image_sign.shape[0:2]
    h1, w1 = image_sign_match.shape[0:2]

    processImage = image_sign.copy()
    for negBox in negBoxes:
        nw_offset = int(0.3 * (negBox[2] - negBox[0]))
        nh_offset = int(0.3 * (negBox[3] - negBox[1]))
        negBox = [max(negBox[0] - nw_offset, 0), max(negBox[1] - nh_offset, 0), min(negBox[2] + nw_offset, w0), min(negBox[3] + nh_offset, h0)]
        processImage[negBox[1]:negBox[3], negBox[0]:negBox[2], :] = 0

    roi = [max(gtBox[0] - w_offset, 0), max(gtBox[1] - h_offset, 0), min(gtBox[2] + w_offset, w0),
              min(gtBox[3] + h_offset, h0)]
    gt_roi = [max(gtBox_match[0] - gtw_offset, 0), max(gtBox_match[1] - gth_offset, 0), min(gtBox_match[2] + gtw_offset, w1),
                   min(gtBox_match[3] + gth_offset, h1)]

    roim = [max(gtBox[0] - wm_offset, 0), max(gtBox[1] - hm_offset, 0), min(gtBox[2] + wm_offset, w0),
              min(gtBox[3] + hm_offset, h0)]
    gt_roim = [max(gtBox_match[0] - gtwm_offset, 0), max(gtBox_match[1] - gthm_offset, 0), min(gtBox_match[2] + gtwm_offset, w1),
                   min(gtBox_match[3] + gthm_offset, h1)]

    queryImage0 = processImage[roi[1]:roi[3], roi[0]:roi[2], :]
    templateImage = image_sign_match[gt_roi[1]:gt_roi[3], gt_roi[0]:gt_roi[2], :]

    queryImage0 = cv2.resize(queryImage0, (64, 64))
    templateImage = cv2.resize(templateImage, (64, 64))
    # queryImage0_m = deHaze(queryImage0 / 255.0) * 255
    # templateImage_m = deHaze(templateImage / 255.0) * 255
    # queryImage0_m = np.array(queryImage0_m, dtype=np.uint8)
    # templateImage_m = np.array(templateImage_m, dtype=np.uint8)
    # cv2.imshow("queryImage0", queryImage0)
    # cv2.imshow("templateImage", templateImage)
    # cv2.imshow("queryImage0_m", queryImage0_m)
    # cv2.imshow("templateImage_m", templateImage_m)
    # cv2.waitKey(0)


    queryImageHash = Image.fromarray(cv2.cvtColor(processImage[roim[1]:roim[3], roim[0]:roim[2], :], cv2.COLOR_BGR2RGB))
    templateImageHash = Image.fromarray(cv2.cvtColor(image_sign_match[gt_roim[1]:gt_roim[3], gt_roim[0]:gt_roim[2], :], cv2.COLOR_BGR2RGB))

    highfreq_factor = 1
    hash_size = 8
    img_size = highfreq_factor * hash_size
    phashQuery = imagehash.phash(queryImageHash, img_size, highfreq_factor)
    phashTemplate = imagehash.phash(templateImageHash, img_size, highfreq_factor)
    whashQuery = imagehash.whash(queryImageHash, img_size)
    whashTemplate = imagehash.whash(templateImageHash, img_size)

    phasgsim = 1 - (phashQuery - phashTemplate) / len(phashTemplate.hash) ** 2
    whasgsim = 1 - (whashQuery - whashTemplate) / len(whashTemplate.hash) ** 2
    hashError = 1 - (phasgsim + whasgsim) / 2.0

    # cv2.imshow('queryImage0', queryImage0)
    # cv2.imshow('templateImage', templateImage)
    # cv2.waitKey(0)
    sift = cv2.xfeatures2d.SIFT_create()
    kp0, des0 = sift.detectAndCompute(queryImage0, None)
    kp, des = sift.detectAndCompute(templateImage, None)
    # print(des0, des1, des)

    error = 1.0
    feature_error = 1.0
    count = 0
    if des0 is not None and des is not None:
        if len(des0) >= 2 and len(des) >= 2:
            FLANN_INDEX_KDTREE = 0
            indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            searchParams = dict(checks=500)
            flann = cv2.FlannBasedMatcher(indexParams, searchParams)
            matches0 = flann.knnMatch(des0, des, k=2)
            matchesMask0 = [[0, 0] for i in range(len(matches0))]

            for i, (m, n) in enumerate(matches0):
                if m.distance < 0.6 * n.distance:
                    matchesMask0[i] = [1, 0]
                    count += 1
            error_pd = matchCenterError(kp0, matchesMask0, matches0, roi, gtBox, mode=0)
            error_match = matchCenterError(kp, matchesMask0, matches0, gt_roi, gtBox_match, mode=1)
            if error_pd >= 1.0:
                error = 1.0
                feature_error = 1.0
            else:
                feature_error = 1.0 - float(count) / len(des0)
                error = abs(error_pd - error_match)
                # print(feature_error, error)
    else:
        feature_error = 1.0
        error = 1.0
    # print(error, feature_error, hashError)
    return error, feature_error, hashError