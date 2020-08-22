# -*- coding: utf-8 -*-
"""
Created on Tues Jun 30 10:20:01 2020

@author: Jingxiao Gu
"""

import cv2
import json
import numpy as np
import os


def parseJson(fullTag, threshold, clsThreshold):
    fJson = open(fullTag)
    fullJson = json.load(fJson)
    preSigns = fullJson['signs']
    postSigns = []

    for sign in preSigns:
        if sign['score'] > threshold:# and sign['cls_score'] > clsThreshold:
            postSigns.append(sign)
        else:
            print(sign['score'])

    fullJson['signs'] = postSigns

    return fullJson


if __name__ == '__main__':
    # fullFolder = "../../output_results/test/ensemble_test/"
    # thresFolder = "../../output_results/test/ensemble_test_050/"
    fullFolder = "../../output_results/val/ensemble_val/"
    thresFolder = "../../output_results/val/ensemble_val_050/"
    threshold = 0.50
    clsThreshold = 0.0

    for tag in os.listdir(fullFolder):
        fullTag = os.path.join(fullFolder, tag)
        thresTag = parseJson(fullTag, threshold, clsThreshold)

        with open(os.path.join(thresFolder, tag), "w") as f:
            json.dump(thresTag, f)