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
import math
import random
import functools
import numpy as np
import paddle
from imgtool import process_image
import glob
import json
import signal
random.seed(0)


def sign2dict(signs, pic_path):
    # sign to dictionary
    # key(string): sign_id
    # value(dict): origin sign
    sign_dict = {}
    for sign in signs:
        if sign['h'] <= 0 or sign['w'] <= 0: continue
        sign_id = sign['sign_id']
        sign['pic_name'] = os.path.join(pic_path, sign['pic_id'] + '.jpg')
        sign_dict[sign_id] = sign
    return sign_dict


def init_sign(mode, path, detect_path=None):
    pic_path = os.path.join(path, 'pic')
    data_type = path.split('/')[-1]
    if detect_path is None:
        data_path = os.path.dirname(path)
        if mode == 'train':
            # tag_path = os.path.join(data_path, 'tag')
            anno_path = os.path.join(path, 'train_tag')
        else:
            anno_path = os.path.join(path, 'val_tag')
        assert os.path.exists(
            anno_path), 'Annotation path: {} does not exist'.format(anno_path)
    else:
        anno_path = detect_path
        assert os.path.exists(
            anno_path), 'detect_path: {} does not exist, please get detection result at first'.format(anno_path)

    if mode == 'train':
        anno_files = glob.glob(os.path.join(anno_path, '*.json'))
        cls_id = 0
        # cls_id to sign annotations
        anno_data = []
        anno_list = []
        for anno_file in anno_files:
            # sign_id to cls_id
            sign_cls = {}
            anno = json.load(open(anno_file))
            signs = anno['signs']
            sign_dict = sign2dict(signs, pic_path)
            match = anno['match']
            for pair in match:
                sign = pair['sign_id']
                match_sign = pair['match_sign_id']
                if 'sign' in sign and sign not in sign_dict.keys():
                    print('Illegal H or W in sign: {} in {} and ignore it'.
                          format(sign, anno_file))
                    continue
                if 'sign' in match_sign and match_sign not in sign_dict.keys():
                    print('Illegal H or W in sign: {} in {} and ignore it'.
                          format(match_sign, anno_file))
                    continue

                if sign in sign_cls.keys():
                    tmp_cls = sign_cls[sign]
                    if 'sign' in match_sign:
                        anno_data[tmp_cls].append(sign_dict[match_sign])
                        sign_cls[match_sign] = tmp_cls
                elif match_sign in sign_cls.keys():
                    tmp_cls = sign_cls[match_sign]
                    anno_data[tmp_cls].append(sign_dict[sign])
                    sign_cls[sign] = tmp_cls
                else:
                    anno_data.append([sign_dict[sign]])
                    sign_cls[sign] = cls_id
                    if 'sign' in match_sign:
                        anno_data[cls_id].append(sign_dict[match_sign])
                        sign_cls[match_sign] = cls_id
                    cls_id += 1
        print('total matched class number: {}'.format(cls_id))
        for cls, anno in enumerate(anno_data):
            for sign in anno:
                anno_list.append((sign, cls))
        print('total instance number: {}'.format(len(anno_list)))
        return anno_data, anno_list

    else:
        anno_files = glob.glob(os.path.join(anno_path, '*.json'))
        group_id = 0
        sign_list = []
        for anno_file in anno_files:
            anno = json.load(open(anno_file))
            assert 'signs' in anno, "'signs' is not in json file: {}, please check path or obtain detection result at first.".format(
                anno_file)
            signs = anno['signs']
            group = anno['group']
            # seq_id: 0 stands for A and 1 stands for B
            for sign in signs:
                pic_name = os.path.join(pic_path, sign['pic_id'] + '.jpg')
                seq_id = int(sign['pic_id'] in group[1]['pic_list'])
                sign.update({
                    'group_id': group_id,
                    'seq_id': seq_id,
                    'pic_name': pic_name
                })
            sign_list.extend(signs)
            group_id += 1
        return sign_list


def get_pos_pair(pos_data_list, data_ind):
    anchor_ind = data_ind[0]
    anchor_id = pos_data_list[anchor_ind]['sign_id']
    for pos_ind in data_ind[1:]:
        pos_id = pos_data_list[pos_ind]['sign_id']
        # find pos and anchor in A and B
        if pos_id[5] != anchor_id[5]:
            return anchor_ind, pos_ind


def triplet_iterator(data, settings):
    batch_size = settings.train_batch_size
    assert (batch_size % 3 == 0)

    def train_iterator():
        total_count = settings.train_batch_size * (settings.total_iter_num + 1)
        count = 0
        lab_num = len(data)
        ind = list(range(0, lab_num))
        while True:
            random.shuffle(ind)
            ind_pos, ind_neg = ind[:2]
            pos_data_list = data[ind_pos]
            if len(pos_data_list) < 2:
                continue
            data_ind = list(range(0, len(pos_data_list)))
            random.shuffle(data_ind)
            # select sign_A and sign_B separately
            anchor_ind, pos_ind = get_pos_pair(pos_data_list, data_ind)

            neg_data_list = data[ind_neg]
            neg_ind = random.randint(0, len(neg_data_list) - 1)
            anchor_path = pos_data_list[anchor_ind]
            # print(anchor_path, pos_ind)
            yield anchor_path, pos_ind
            pos_path = pos_data_list[pos_ind]
            yield pos_path, pos_ind
            neg_path = neg_data_list[neg_ind]
            yield neg_path, neg_ind
            count += 3
            if count >= total_count:
                return

    return train_iterator


def arcmargin_iterator(data, settings):
    def train_iterator():
        total_count = settings.train_batch_size * (settings.total_iter_num + 1)
        count = 0
        while True:
            for items in data:
                sign, label = items
                yield sign, label
                count += 1
                if count >= total_count:
                    return

    return train_iterator


def image_iterator(data):
    def pair_iterator():
        for items in data:
            group = items['group_id']
            label = int(items['type'])
            seq_id = int(items['seq_id'])
            yield items, group, label, seq_id

    return pair_iterator


def createreader(settings, mode):
    def metric_reader():
        if mode == 'train':
            train_data, train_image_list = init_sign(mode, settings.data_path)
            loss_name = settings.loss_name
            if loss_name in ["softmax", "arcmargin"]:
                return arcmargin_iterator(train_image_list, settings)()
            elif loss_name == 'triplet':
                return triplet_iterator(train_data, settings)()
            else:
                raise NameError(
                    "Invalid loss name: {}. You should use softmax, arcmargin, triplet".
                    format(loss_name))
        else:
            sign_list = init_sign(mode, settings.data_path,
                                  settings.detect_path)
            return image_iterator(sign_list)()

    image_shape = settings.image_shape.split(',')
    assert (image_shape[1] == image_shape[2])
    image_size = int(image_shape[2])
    image_mapper = functools.partial(
        process_image,
        mode=mode,
        color_jitter=False,
        rotate=False,
        crop_size=image_size)

    reader = paddle.reader.xmap_readers(
        image_mapper, metric_reader, 8, 1000, order=True)
    return reader


def train(settings):
    return createreader(settings, "train")


def test(settings):
    return createreader(settings, "test")
