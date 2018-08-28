import os
import numpy as np
from glob import glob
from image_tools import img_transform


def remain_time(epoch_now, epoch_total, batch_now, batch_total, pass_time):
    step = epoch_now * batch_total + batch_now
    total_step = epoch_total * batch_total
    return pass_time * (total_step - step) / step


def copy_img(train_size, test_size, src_dir, dst_dir='./img_data/lsun_bedroom'):
    # src_dir = r'C:\Users\wangboquan\Personal\Projects\dataset\lsun_bedroom'
    # dst_dir = './img_data/lsun_bedroom'

    old_train_data = glob('{}/train/*.png'.format(dst_dir))
    old_test_data = glob('{}/test/*.png'.format(dst_dir))
    if len(old_train_data) == 0 and len(old_test_data) == 0:
        print("[ Distance dir is empty ]")
    else:
        for item in old_train_data:
            os.remove(item)
        for item in old_test_data:
            os.remove(item)
        print("[ Cleaning distance dir ]")

    print("[ Searching... ]")
    files = np.random.choice(glob('{}/*.webp'.format(src_dir)), train_size+test_size)

    if not os.path.exists(os.path.join(dst_dir, 'train')):
        os.makedirs(os.path.join(dst_dir, 'train'))
    if not os.path.exists(os.path.join(dst_dir, 'test')):
        os.makedirs(os.path.join(dst_dir, 'test'))

    for idx, file in enumerate(files):
        (file_path, file_name) = os.path.split(file)
        (file_name, file_ext) = os.path.splitext(file_name)
        if idx < train_size:
            img_transform(file).save(os.path.join(dst_dir, "train", file_name + '.png'), 'PNG')
        else:
            img_transform(file).save(os.path.join(dst_dir, "test", file_name + '.png'), 'PNG')
        print("%s (%s/%s)" % (file, idx + 1, train_size+test_size))