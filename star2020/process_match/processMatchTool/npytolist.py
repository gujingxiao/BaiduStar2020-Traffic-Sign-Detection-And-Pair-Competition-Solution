# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:29:01 2020

@author: DYP
"""

import numpy as np
import glob
import json
import os


def generate_sign_id(anno, filename):
    group = anno['group']
    group_a = group[0]['pic_list']
    group_b = group[1]['pic_list']
    signs = anno['signs']
    id_a, id_b = 1, 1
    for sign in signs:
        if sign['pic_id'] in group_a:
            sign['sign_id'] = 'sign_A_{}'.format(id_a)
            id_a += 1
        elif sign['pic_id'] in group_b:
            sign['sign_id'] = 'sign_B_{}'.format(id_b)
            id_b += 1
        else:
            print('illegal pic id: {} in json file: {}'.format(sign['pic_id'],
                                                               filename))
    return anno

def feature2json(results, groups, labels, seq_id, match_path, save_path):
    # windows
    # json_list = np.load('test_npy/test_list.npy')
    # ubuntu 
    json_list = glob.glob(os.path.join(match_path, '*.json'))
    #----------------------------------------------------
    
    group_num = np.max(groups) + 1
    res_group = [[] for i in range(group_num)]
    im_id = 0
    for res1, res2, res3, res4, res5, l, s, g in zip(results[0], results[1], results[2], results[3], results[4], labels, seq_id, groups):
        res_group[g].append([res1, res2, res3, res4, res5, l, s, im_id])
        im_id += 1
    
    for i, anno_file in enumerate(json_list):
        print(i)
        anno = json.load(open(anno_file))
        anno = generate_sign_id(anno, anno_file)
        signs = anno['signs']
        res_featers = res_group[i]
        new_signs = []
        for featuer, sign in zip(res_featers, signs):
            sign['feature1'] = featuer[0].tolist()
            sign['feature2'] = featuer[1].tolist()
            sign['feature3'] = featuer[2].tolist()
            sign['feature4'] = featuer[3].tolist()
            sign['feature5'] = featuer[4].tolist()
            new_signs.append(sign)
        anno.update({'signs': new_signs})
        result_file = os.path.join(save_path, os.path.basename(anno_file))
        with open(result_file, 'w') as fp:
            json.dump(anno, fp)


if __name__ == '__main__':
    # 在匹配网络的eval中save_npy设置为true 保存g s l f
    match_path = '../../../output_results/test/ensemble_test_match'  # 匹配网络地址
    save_path = '../../../output_results/test/feature_test' # 特征存放地址

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    g = np.load('../../../output_results/test/ensemble_test_match_g.npy')
    s = np.load('../../../output_results/test/ensemble_test_match_s.npy')
    l = np.load('../../../output_results/test/ensemble_test_match_l.npy')
    f1 = np.load('../../../output_results/test/ensemble_test_match_ResNet50.npy')[:,:,0,0]
    f2 = np.load('../../../output_results/test/ensemble_test_match_SE_ResNeXt50_vd_32x4d.npy')[:, :, 0, 0]
    f3 = np.load('../../../output_results/test/ensemble_test_match_ResNeXt152_vd_64x4d.npy')[:, :, 0, 0]
    f4 = np.load('../../../output_results/test/ensemble_test_match_ResNeXt152_vd_32x4d.npy')[:, :, 0, 0]
    f5 = np.load('../../../output_results/test/ensemble_test_match_ResNeXt101_vd_64x4d.npy')[:, :, 0, 0]

    # match_path = '../../../output_results/val/ensemble_val_050_match'  # 匹配网络地址
    # save_path = '../../../output_results/val/feature_val'  # 特征存放地址
    #
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_path)
    #
    # g = np.load('../../../output_results/val/ensemble_val_050_match_g.npy')
    # s = np.load('../../../output_results/val/ensemble_val_050_match_s.npy')
    # l = np.load('../../../output_results/val/ensemble_val_050_match_l.npy')
    # f1 = np.load('../../../output_results/val/ensemble_val_050_match_ResNet50.npy')[:, :, 0, 0]
    # f2 = np.load('../../../output_results/val/ensemble_val_050_match_SE_ResNeXt50_vd_32x4d.npy')[:, :, 0, 0]
    # f3 = np.load('../../../output_results/val/ensemble_val_050_match_ResNeXt152_vd_64x4d.npy')[:, :, 0, 0]
    # f4 = np.load('../../../output_results/val/ensemble_val_050_match_ResNeXt152_vd_32x4d.npy')[:, :, 0, 0]
    # f5 = np.load('../../../output_results/val/ensemble_val_050_match_ResNeXt101_vd_64x4d.npy')[:, :, 0, 0]

    f = [f1, f2, f3, f4, f5]
    
    feature2json( f, g, l, s, match_path, save_path)

