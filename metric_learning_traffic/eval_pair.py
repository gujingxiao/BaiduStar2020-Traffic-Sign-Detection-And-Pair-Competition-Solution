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
import sys
import math
import time
import argparse
import functools
import numpy as np
import paddle
import paddle.fluid as fluid
import models
import reader
from utility import add_arguments, print_arguments, check_cuda, load_pretrain
from utility import fmt_time, recall_topk, post_process, save_result

# yapf: disable
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('batch_size', int, 128, "Minibatch size.")
add_arg('image_shape', str, "3,96,96", "Input image size.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('data_path', str, "../../data/traffic_data/test", "path of validation data or test data")
add_arg('thresh', float, 0.5, "threshold for similarity distance")
add_arg('top_k', int, 10, "the number of images to match")
add_arg('output_path', str, "./output/cascade101_c3_c19_result_040", "path for saving json result")
add_arg('detect_path', str, None, "path of detection result")
add_arg('save_npy', bool, True, "path of detection result")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    # parameters from arguments
    model_name = args.model
    pretrained_model = args.pretrained_model
    save_npy = args.save_npy
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.data(
        name='image', shape=[None] + image_shape, dtype='float32')
    group = fluid.data(name='group', shape=[None, 1], dtype='int64')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')
    seq_id = fluid.data(name='seq_id', shape=[None, 1], dtype='int64')

    test_loader = fluid.io.DataLoader.from_generator(
        feed_list=[image, group, label, seq_id],
        capacity=64,
        use_double_buffer=True,
        iterable=True)

    # model definition
    model = models.__dict__[model_name]()
    out = model.net(input=image, embedding_size=args.embedding_size)

    test_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        fluid.load(
            program=test_program, model_path=pretrained_model, executor=exe)

    test_loader.set_sample_generator(
        reader.test(args),
        batch_size=args.batch_size,
        drop_last=False,
        places=place)

    fetch_list = [out.name]

    f, l, g, s = [], [], [], []
    for batch_id, data in enumerate(test_loader()):
        t1 = time.time()
        [feas] = exe.run(test_program, fetch_list=fetch_list, feed=data)
        group = np.asarray(data[0]['group'])
        label = np.asarray(data[0]['label'])
        seq_id = np.asarray(data[0]['seq_id'])
        f.append(feas)
        g.append(np.squeeze(group))
        l.append(np.squeeze(label))
        s.append(np.squeeze(seq_id))

        t2 = time.time()
        period = t2 - t1
        if batch_id % 2 == 0:
            print("[%s] testbatch %d, time %2.2f sec" % \
                    (fmt_time(), batch_id, period))

    f = np.vstack(f)
    g = np.hstack(g)
    l = np.hstack(l)
    s = np.hstack(s)

    if save_npy == True:
        np.save(args.output_path + '_' + model_name, f)
        np.save(args.output_path + '_' + 'g', g)
        np.save(args.output_path + '_' + 'l', l)
        np.save(args.output_path + '_' + 's', s)

    res_final, res_score = post_process(f, g, l, s, args.thresh, k=args.top_k)

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print("Saving result to {}".format(output_path))
    save_result(res_final, res_score, output_path, args.detect_path)
    sys.stdout.flush()


def main():
    args = parser.parse_args()
    print_arguments(args)
    check_cuda(args.use_gpu)
    eval(args)


if __name__ == '__main__':
    main()
