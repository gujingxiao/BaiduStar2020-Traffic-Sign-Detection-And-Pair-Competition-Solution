# -*- coding: utf-8 -*-
"""
    @Author  : Jingxiao Gu
    @Time    : 2020/06/21
"""
import os
import random

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

if __name__ == '__main__':
    trainFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/final_classification/train/"
    valFolder = "/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/final_classification/val/"

    train_list = []
    val_list = []

    for index, subfolder in enumerate(os.listdir(trainFolder)):
        total_list = os.listdir(os.path.join(trainFolder, subfolder))
        random.shuffle(total_list)

        sub_length = len(total_list)
        for item in total_list:
            label = labels[subfolder]
            dir = 'train/' + subfolder + '/' + item

            if sub_length > 10 and sub_length <= 50:
                for iii in range(20):
                    train_list.append(dir + ' ' + str(label) + '\n')
            elif sub_length > 50 and sub_length <= 100:
                for iii in range(13):
                    train_list.append(dir + ' ' + str(label) + '\n')
            elif sub_length > 100 and sub_length <= 200:
                for iii in range(8):
                    train_list.append(dir + ' ' + str(label) + '\n')
            elif sub_length > 200 and sub_length <= 500:
                for iii in range(5):
                    train_list.append(dir + ' ' + str(label) + '\n')
            elif sub_length > 500 and sub_length <= 1000:
                for iii in range(3):
                    train_list.append(dir + ' ' + str(label) + '\n')
            else:
                train_list.append(dir + ' ' + str(label) + '\n')

    for index, subfolder in enumerate(os.listdir(valFolder)):
        total_list = os.listdir(os.path.join(valFolder, subfolder))
        for item in total_list:
            label = labels[subfolder]
            dir = 'val/' + subfolder + '/' + item
            val_list.append(dir + ' ' + str(label) + '\n')

    random.shuffle(train_list)

    print(len(train_list), len(val_list))

    with open("/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/final_classification//train_list.txt", "w") as f:
        f.writelines(train_list)
    f.close()

    with open("/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/final_classification/val_list.txt", "w") as f:
        f.writelines(val_list)
    f.close()