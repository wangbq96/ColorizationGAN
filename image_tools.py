import os
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
from PIL import Image


def cvtRGB2YUV(image):
    cvt_matrix = np.array([[0.299, -0.169, 0.5], [0.587, -0.331, -0.419], [0.114, 0.5, -0.081]], dtype=np.float32)
    return image.dot(cvt_matrix) + [0, 127.5, 127.5]


def cvtYUV2RGB(image):
    cvt_matrix = np.array([[1, 1, 1], [-0.00093, -0.3437, 1.77216], [1.401687, -0.71417, 0.00099]], dtype=np.float32)
    return (image - [0, 127.5, 127.5]).dot(cvt_matrix).clip(min=0, max=255)


def load_images(image_path):
    input_img = cvtRGB2YUV(imread(image_path).astype(np.float))
    img_a = input_img[:, :, 0:1]
    img_b = input_img

    img_a = img_a / 127.5 - 1.
    img_b = img_b / 127.5 - 1.

    img_ab = np.concatenate((img_a, img_b), axis=2)
    return img_ab


def save_images(images, image_dir, is_real):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    images = (images + 1.) * 127.5
    if is_real:
        image_type = 'real'
    else:
        image_type = 'fake'
    counter = 0
    for idx, item in enumerate(images):
        item = cvtYUV2RGB(item)
        imsave(os.path.join(image_dir, '{}_{}.png'.format(image_type, idx)), item)
        counter += 1


def img_transform(file_name, img_size=256):
    old_img = Image.open(file_name, 'r')
    h = old_img.size[1]
    w = old_img.size[0]
    x = (w - h) / 2
    box = (x, 0, x+h, h)
    return old_img.crop(box).resize((img_size, img_size))


def create_gray(src_dir='./test', dir_num=64, pair_num=16):
    for dir_idx in range(dir_num):
        for pair_idx in range(pair_num):
            color_file = os.path.join(src_dir, str(dir_idx), 'real_{}.png'.format(pair_idx))
            color_img = cvtRGB2YUV(imread(color_file).astype(np.float))
            img_y = color_img[:, :, 0:1]
            img_uv = np.ones((256, 256, 2)) * 128
            gray_img = cvtYUV2RGB(np.concatenate((img_y, img_uv), axis=2))
            gray_file = os.path.join(src_dir, str(dir_idx), 'gray_{}.png'.format(pair_idx))
            print(color_file + ' -> ' + gray_file)
            imsave(gray_file, gray_img)





