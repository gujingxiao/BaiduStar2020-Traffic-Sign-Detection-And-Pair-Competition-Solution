# -*- coding: utf-8 -*-
"""
@author: Jingxiao Gu
"""
import os
import numpy as np
import json

from star2020.process_match.processMatchTool.cascadeFilter import correctWrongOrder, deleteWrongTypeMatch, filterThresholdMatch,\
    deleteUnreasonableMatch, filterMultiToOneMatch, filterOneToMultiMatch, getNonMatchBack, getLostMatchBack
from star2020.process_match.processMatchTool.matchUtils import match_split


def postProcessSignMatchs(prePath, featurePath, picFolder, mode, imageResolutions, detThreshold, mtThreshold, postMethod):
    # Load json and get all values
    fJson = open(prePath)
    preJson = json.load(fJson)
    preMatchs = preJson['match']
    preSigns = preJson['signs']
    feature_data = json.load(open(featurePath))
    feature_signs = feature_data['signs']
    match_a, match_b = match_split(preMatchs)

    # Regular Check and Filter
    pdMatchs = correctWrongOrder(preMatchs)
    pdMatchs = deleteWrongTypeMatch(pdMatchs, preSigns)

    mt_thres = mtThreshold
    dt_thres = detThreshold
    pdMatchs, pdSigns = filterThresholdMatch(pdMatchs, preSigns, det_thres=dt_thres, mt_thres=mt_thres)
    preJson['match'] = pdMatchs
    preJson['signs'] = pdSigns

    # Delete some unreasonable matches based on single spatial information
    filterMatchs, allInfoMatchs = deleteUnreasonableMatch(preJson, imageResolutions)
    preJson['match'] = filterMatchs

    # Filter multiToOne and oneToMulti match based on a single image pair
    confidenceRange = 0.01
    singleMatchs, keepDuplicateds, noMatchs = filterMultiToOneMatch(preJson, allInfoMatchs, postMethod, picFolder, confidenceRange)

    keepDuplicateds = keepDuplicateds + singleMatchs
    newsingleMatchs, newkeepDuplicateds = filterOneToMultiMatch(preJson, keepDuplicateds, postMethod, picFolder, confidenceRange)
    newMatchs = newkeepDuplicateds + noMatchs + newsingleMatchs

    # Since many matches have been removed, we need to get those "Non match sign" back because they are indeed detected with high confidence.
    preJson['match'] = newMatchs
    preJson['match'] = getLostMatchBack(preJson, newMatchs, picFolder, feature_signs, match_a)
    preJson['match'] = getNonMatchBack(preJson, preJson['match'])

    # Make pre to post, all done
    postJson = preJson

    # For test, delete unnecessary keys
    if mode == 'test':
        for item in postJson['signs']:
            item.pop('score')

        for item in postJson['match']:
            item.pop('match_score')
            item.pop('sign_pic')
            item.pop('match_pic')
            item.pop('sign_type')
            item.pop('match_type')

    return postJson


# Post processing based on some general rules
if __name__ == '__main__':
    postMethod = 'easy' # easy or combine

    # val
    # detThreshold = 0.5
    # preFolder = "../../../output_results/val/ensemble_val_050_match/"
    # postFolder = "../../../output_results/val/ensemble_val_050_match_check/"
    # resolution = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/valResolution.json"
    # picFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/pic/"
    # feature_path = '../../../output_results/val/feature_val' # 特征文件目录，特征文件由npytojson生成
    # mode = 'val'

    # test
    detThreshold = 0.65
    preFolder = "../../../output_results/test/ensemble_test_match_backup_for_final/"
    postFolder = "../../../output_results/test/ensemble_test_match_check_065_020_final/"
    resolution = "../../../output_results/tag/testResolution.json"
    picFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test/pic/"
    feature_path = '../../../output_results/test/feature_test'  # 特征文件目录，特征文件由npytojson生成
    mode = 'test'

    mt_thres = [0.25] * 19
    mt_thres[18] = 0.2
    dt_thres = [detThreshold] * 19
    # dt_thres[0] = 0.6
    # dt_thres[1] = 0.55
    # dt_thres[2] = 0.55
    # dt_thres[3] = 0.6
    # dt_thres[4] = 0.6
    # dt_thres[5] = 0.55
    # dt_thres[6] = 0.65
    # dt_thres[7] = 0.6
    # dt_thres[8] = 0.6
    # dt_thres[9] = 0.65
    # dt_thres[10] = 0.65
    # dt_thres[11] = 0.55
    # dt_thres[12] = 0.55
    # dt_thres[13] = 0.65
    # dt_thres[14] = 0.65
    # dt_thres[15] = 0.6
    # dt_thres[16] = 0.6
    # dt_thres[17] = 0.55

    rJson = open(resolution)
    imageResolutions = json.load(rJson)

    for index, tag in enumerate(os.listdir(preFolder)):
        print(index, tag)
        prePath = os.path.join(preFolder, tag)
        postPath = os.path.join(postFolder, tag)
        featurePath = os.path.join(feature_path, tag)
        postJson = postProcessSignMatchs(prePath, featurePath, picFolder, mode, imageResolutions, dt_thres, mt_thres, postMethod)

        with open(postPath, "w") as f:
            json.dump(postJson, f)
