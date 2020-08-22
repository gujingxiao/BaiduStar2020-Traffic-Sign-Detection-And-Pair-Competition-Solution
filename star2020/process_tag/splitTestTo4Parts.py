import os
import shutil

# test_input_folder = '/media/gujingxiao/f577505e-73a2-41d0-829c-eb4d01efa827/BaiduStar2020/test/input'
test_input_folder = '../../output_results/test/ensemble_test_c3_070'

test_input_list = os.listdir(test_input_folder)
length = len(test_input_list)
offset = int(length / 4)
test_fold1 = test_input_list[0:offset]
test_fold2 = test_input_list[offset: offset + offset]
test_fold3 = test_input_list[offset + offset : offset + offset + offset]
test_fold4 = test_input_list[offset + offset + offset:]

for fold in test_fold1:
    shutil.copy(os.path.join(test_input_folder, fold), os.path.join(test_input_folder + '1', fold))

for fold in test_fold2:
    shutil.copy(os.path.join(test_input_folder, fold), os.path.join(test_input_folder + '2', fold))

for fold in test_fold3:
    shutil.copy(os.path.join(test_input_folder, fold), os.path.join(test_input_folder + '3', fold))

for fold in test_fold4:
    shutil.copy(os.path.join(test_input_folder, fold), os.path.join(test_input_folder + '4', fold))
