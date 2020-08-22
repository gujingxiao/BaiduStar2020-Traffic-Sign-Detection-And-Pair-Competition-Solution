# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import tools.infer.utils as utils
import argparse
import os
import numpy as np
import json
import cv2

import paddle.fluid as fluid

from ppcls.modeling import architectures


labels = {
    0: '102',
    1: '103',
    2: '104',
    3: '105',
    4: '106',
    5: '107',
    6: '108',
    7: '109',
    8: '110',
    9: '111',
    10: '112',
    11: '113',
    12: '201',
    13: '202',
    14: '203',
    15: '204',
    16: '205',
    17: '206',
    18: '207',
    19: '208',
}

def create_predictor(model_type, pretrained_model):
    def create_input():
        image = fluid.data(
            name='image', shape=[None, 3, 160, 160], dtype='float32')
        return image

    def create_model(model_type, model, input, class_dim=2):
        if model_type == "GoogLeNet":
            out, _, _ = model.net(input=input, class_dim=class_dim)
        else:
            out = model.net(input=input, class_dim=class_dim)
            out = fluid.layers.softmax(out)
        return out

    model = architectures.__dict__[model_type]()

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()

    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            image = create_input()
            out = create_model(model_type, model, image)

    infer_prog = infer_prog.clone(for_test=True)
    fluid.load(program=infer_prog, model_path=pretrained_model, executor=exe)

    return exe, infer_prog, [image.name], [out.name]


def create_operators():
    size = 160
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    resize_op = utils.ResizeImage(resize_short=160)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [resize_op, crop_op, normalize_op, totensor_op]


def preprocess(data, ops):
    data = data[:, :, ::-1]
    for op in ops:
        data = op(data)

    return data


def postprocess(outputs, topk=1):
    output = outputs[0]
    prob = np.array(output).flatten()
    index = prob.argsort(axis=0)[-topk:][::-1].astype('int32')
    return index, prob[index]


def main():
    # det_result_path = '../output_results/test/ensemble_test_050/'
    # save_class_result_path = '../output_results/test/ensemble_test_050_filter/'
    # data_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test/pic/'

    det_result_path = '../output_results/val/ensemble_val_050_filter_class/'
    save_class_result_path = '../output_results/val/ensemble_val_050_filter_class/'
    data_path = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/train/pic/'

    model_type = 'Res2Net101_vd_26w_4s'
    pretrained_model = 'output/Res2Net101_vd_26w_4s/false_best_model_160/ppcls'
    operators = create_operators()
    exe, program, feed_names, fetch_names = create_predictor(model_type, pretrained_model)
    for idddd, tag in enumerate(os.listdir(det_result_path)):
        tagPath = os.path.join(det_result_path, tag)
        fJson = open(tagPath)
        preJson = json.load(fJson)

        for index, seq0_pic in enumerate(preJson['group'][0]['pic_list']):
            # print(seq0_pic)
            image = cv2.imread(os.path.join(data_path, seq0_pic + '.jpg'))

            for sign in preJson['signs']:
                if sign['pic_id'] == seq0_pic:
                    if sign['type'] != '301':
                        if sign['w'] > 20:
                            xmin = max(0, sign['x'] - int(sign['w'] * 0.05))
                            xmax = min(image.shape[1], sign['x'] + sign['w'] + int(sign['w'] * 0.05))
                        else:
                            xmin = max(0, sign['x'] - 1)
                            xmax = min(image.shape[1], sign['x'] + sign['w'] + 1)

                        if sign['h'] > 20:
                            ymin = max(0, sign['y'] - int(sign['h'] * 0.05))
                            ymax = min(image.shape[1], sign['y'] + sign['h'] + int(sign['h'] * 0.05))
                        else:
                            ymin = max(0, sign['y'] - 1)
                            ymax = min(image.shape[0], sign['y'] + sign['h'] + 1)

                        roi_image = image[int(ymin):int(ymax), int(xmin):int(xmax), :]

                        data = preprocess(roi_image, operators)
                        data = np.expand_dims(data, axis=0)

                        outputs = exe.run(program,
                                          feed={feed_names[0]: data},
                                          fetch_list=fetch_names,
                                          return_numpy=False)

                        idx, prob = postprocess(outputs)
                        if (idx[0] == 11 or idx[0] == 19) and prob[0] > 0.8:
                            sign['type'] = '400'

        for index, seq1_pic in enumerate(preJson['group'][1]['pic_list']):
            # print(seq1_pic)
            image = cv2.imread(os.path.join(data_path, seq1_pic + '.jpg'))

            for sign in preJson['signs']:
                if sign['pic_id'] == seq1_pic:
                    if sign['type'] != '301':
                        if sign['w'] > 20:
                            xmin = max(0, sign['x'] - int(sign['w'] * 0.05))
                            xmax = min(image.shape[1], sign['x'] + sign['w'] + int(sign['w'] * 0.05))
                        else:
                            xmin = max(0, sign['x'] - 1)
                            xmax = min(image.shape[1], sign['x'] + sign['w'] + 1)

                        if sign['h'] > 20:
                            ymin = max(0, sign['y'] - int(sign['h'] * 0.05))
                            ymax = min(image.shape[1], sign['y'] + sign['h'] + int(sign['h'] * 0.05))
                        else:
                            ymin = max(0, sign['y'] - 1)
                            ymax = min(image.shape[0], sign['y'] + sign['h'] + 1)

                        roi_image = image[int(ymin):int(ymax), int(xmin):int(xmax), :]

                        data = preprocess(roi_image, operators)
                        data = np.expand_dims(data, axis=0)

                        outputs = exe.run(program,
                                          feed={feed_names[0]: data},
                                          fetch_list=fetch_names,
                                          return_numpy=False)

                        idx, prob = postprocess(outputs)
                        if (idx[0] == 11 or idx[0] == 19) and prob[0] > 0.8:
                            sign['type'] = '400'

        postSigns = []
        for sign in preJson['signs']:
            if sign['type'] != '400':
                postSigns.append(sign)
            else:
                print(sign['type'])
        preJson['signs'] = postSigns

        with open(os.path.join(save_class_result_path, tag), "w") as f:
            json.dump(preJson, f)
        if idddd % 10 == 0:
            print(idddd)

if __name__ == "__main__":
    main()
