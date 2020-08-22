#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import json
import sys
import argparse
import functools
import numpy as np
import six
import distutils.util

def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def cosine_distance(features, seq_id, thresh, k, weights):
    # Ensemble Distance
    d = 0
    seq_scores = []
    for ensem_id in range(len(weights)):
        fea = []
        for index in range(len(features)):
            fea.append(features[index][ensem_id])
        fea = np.array(fea)
        fea = fea.reshape(fea.shape[0], -1)

        a = np.linalg.norm(fea, axis=1).reshape(-1, 1)

        d0 = 1 - np.dot(fea, fea.T) / (a * a.T)
        d0 = d0 + np.eye(len(fea)) * 1e8
        seq_scores.append(d0)
        if ensem_id == 0:
            d = d0 * weights[ensem_id]
        else:
            d = d + d0 * weights[ensem_id]
    sorted_index = np.argsort(d, 1)
    matched_index = []
    matched_score = []
    k = min(len(fea), k)
    for i in range(len(fea)):
        matched_index.append([])
        matched_score.append([])
        matched = 0
        for j in range(k):
            # avoid matched index with same seq_id
            if seq_id[i] == seq_id[sorted_index[i, j]]: continue
            # print(d[i, sorted_index[i, j]])
            if d[i, sorted_index[i, j]] < thresh:
                flag = 0
                for seq_s in seq_scores:
                    if seq_s[i, sorted_index[i, j]] >= 0.22:
                        flag += 1
                if flag >= 2:
                    print(seq_scores[0][i, sorted_index[i, j]], seq_scores[1][i, sorted_index[i, j]], seq_scores[2][i, sorted_index[i, j]],
                          seq_scores[3][i, sorted_index[i, j]], seq_scores[4][i, sorted_index[i, j]])
                else:
                    matched_index[i].append(sorted_index[i, j])
                    matched_score[i].append(d[i, sorted_index[i, j]])
                    matched = 1
        if not matched:
            matched_index[i].append(-1)
            matched_score[i].append(-1.0000000)
    return matched_index, matched_score


def post_process(features, weights, groups, labels, seq_id, thresh, k):
    group_num = np.max(groups) + 1
    res_group = [[] for i in range(group_num)]
    res_final = [[] for i in range(group_num)]
    res_score = [[] for i in range(group_num)]
    im_id = 0
    for res, l, s, g in zip(features, labels, seq_id, groups):
        res_group[g].append([res, l, s, im_id])
        im_id += 1

    for group_id, res_pg in enumerate(res_group):
        print(group_id)
        res_list = []
        seq_list = []
        if len(res_pg) == 0: continue
        for res in res_pg:
            res_list.append(res[0])
            seq_list.append(res[2])
        matched_index, matched_score = cosine_distance(res_list, seq_list, thresh, k, weights)
        for i, m_idxs in enumerate(matched_index):
            for ii, m_i in enumerate(m_idxs):
                if m_i == -1:
                    res_final[group_id].append({i})
                    res_score[group_id].append(1.0)
                    continue
                if {i, m_i} not in res_final[group_id]:
                    res_final[group_id].append({i, m_i})
                    res_score[group_id].append(matched_score[i][ii])
    # print(res_score)
    return res_final, res_score


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

def save_result(res_final, res_score, output_path, detect_path):
    # convert type to interger for evaluation
    convert_list = ['x', 'y', 'w', 'h']
    anno_files = glob.glob(os.path.join(detect_path, '*.json'))
    for i, anno_file in enumerate(anno_files):
        anno = json.load(open(anno_file))
        anno = generate_sign_id(anno, anno_file)
        signs = anno['signs']
        res = res_final[i]
        score = res_score[i]
        match_list = []
        for sign in signs:
            # sign.pop('score')
            for k in sign.keys():
                if k in convert_list:
                    sign[k] = int(sign[k])
        for j, match_pair in enumerate(res):
            match_pair = list(match_pair)
            score = list(score)
            if 'sign_id' not in signs[match_pair[0]]:
                continue
            if len(match_pair) > 1 and 'sign_id' not in signs[match_pair[1]]:
                continue
            sign_id = signs[match_pair[0]]['sign_id']
            match_list.append({'sign_id': sign_id})
            if len(match_pair) > 1:
                match_sign_id = signs[match_pair[1]]['sign_id']
                match_score = score[j]
            else:
                match_sign_id = " "
                match_score = score[j]
            # print(match_score)
            match_list[-1].update({'match_sign_id': match_sign_id, 'match_score': match_score})
        anno.update({'match': match_list})
        file_name = os.path.split(anno_file)[1]
        result_file = os.path.join(output_path, file_name)

        with open(result_file, 'w') as fp:
            json.dump(anno, fp)
        fp.close()

# yapf: disable
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('thresh', float, 0.2, "threshold for similarity distance")
add_arg('top_k', int, 50, "the number of images to match")
# add_arg('output_path', str, "../../output_results/val/ensemble_val_050_match", "path for saving json result")
# add_arg('detect_path', str, "../../output_results/val/ensemble_val_050_filter", "path of detection result")
add_arg('output_path', str, "../../output_results/test/ensemble_test_match", "path for saving json result")
add_arg('detect_path', str, "../../output_results/test/ensemble_test_050_filter", "path of detection result")


def eval(args):
    BASEfOLDER = '../../output_results/test/ensemble_test_match_'
    # BASEfOLDER = '../../output_results/val/ensemble_val_050_match_'
    f50 = np.load(BASEfOLDER + 'ResNet50.npy')
    fx101 = np.load(BASEfOLDER + 'ResNeXt101_vd_64x4d.npy')
    fsex50 = np.load(BASEfOLDER + 'SE_ResNeXt50_vd_32x4d.npy')
    fx152 = np.load(BASEfOLDER + 'ResNeXt152_vd_32x4d.npy')
    fx152_64 = np.load(BASEfOLDER + 'ResNeXt152_vd_64x4d.npy')
    weights = [1.0 / 5.0] * 5
    # weights = [1.0 / 1.0]
    features = []
    for index in range(f50.shape[0]):
        # features.append([fx152_64[index]])
        features.append([f50[index], fx101[index], fsex50[index], fx152[index], fx152_64[index]])
    g = np.load(BASEfOLDER + 'g.npy')
    l = np.load(BASEfOLDER + 'l.npy')
    s = np.load(BASEfOLDER + 's.npy')

    res_final, res_score = post_process(features, weights, g, l, s, args.thresh, k=args.top_k)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Saving result to {}".format(output_path))
    save_result(res_final, res_score, output_path, args.detect_path)
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
