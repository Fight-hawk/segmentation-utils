import os
import time
import os.path as osp
from random import sample
from multiprocessing import Pool

import cv2
import math
import numpy as np
from tqdm import tqdm

NUM_BGS = 10
BG_FOLDER = r"/home/ubuntu/data/backgrounds"
FG_FOLDER = r"/home/ubuntu/data/images"
A_FOLDER = r"/home/ubuntu/data/alphas"
OUT_FOLDER = r'/home/ubuntu/data/matting-dataset'


def get_bg_files():
    bg_files = []
    for file in os.listdir(BG_FOLDER):
        bg_files.append(osp.join(BG_FOLDER, file))
    return bg_files


BG_FILES = None#get_bg_files()


def merge_fg(fg_folders):
    """
    合并前景文件夹
    :param fg_folders: list, 前景文件夹路径
    :return: list, 前景文件路径
    """
    pass


def mkdir(directory):
    """
    创建目录
    :param directory: str, 目录
    :return: str, 目录
    """
    if not osp.exists(directory):
        os.makedirs(directory)
    return directory


def composite4(fg, bg, a, w, h):
    """
    合成图片
    :param fg: 前景图
    :param bg: 背景图
    :param a:  matte图
    :param w:  图片宽度
    :param h:  图片高度
    :return: 合成图
    """
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def process(im_name, a_path, bg_path, bg_count):
    """
    合成前景和背景图片并保存
    :param bg_path: 背景图片路径
    :param a_path: alpha图片路径
    :param im_name: 前景图片名
    :param bg_count: 背景图编号
    :return:
    """
    im = cv2.imread(osp.join(FG_FOLDER, im_name))
    a = cv2.imread(a_path, 0)
    h, w = im.shape[:2]
    bg = cv2.imread(osp.join(BG_FOLDER, bg_path))
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv2.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv2.INTER_CUBIC)

    if im is None or bg is None or a is None:
        return

    out = composite4(im, bg, a, w, h)
    image_name = osp.join(OUT_FOLDER, 'image', im_name[:-4] + '_' + str(bg_count) + '.jpg')
    alpha_name = osp.join(OUT_FOLDER, 'matte', im_name[:-4] + '_' + str(bg_count) + '.png')
    cv2.imwrite(image_name, out)
    cv2.imwrite(alpha_name, a)


def process_one_fg(fg_name):
    bg_files = sample(BG_FILES, NUM_BGS)
    a_path = osp.join(A_FOLDER, fg_name)
    for i, bg_name in enumerate(bg_files):
        process(fg_name, a_path, bg_name, i + 1)


def do_composite():
    print('Doing composite data...')
    fg_files = os.listdir(FG_FOLDER)
    num_samples = len(fg_files) * NUM_BGS
    print('num_samples: ' + str(num_samples))

    start = time.time()

    with Pool(processes=4) as p:
        max_ = len(fg_files)
        print('num_fg_files: ' + str(max_))
        with tqdm(total=max_) as pbar:
            for i, _ in tqdm(enumerate(p.imap_unordered(process_one_fg, fg_files))):
                pbar.update()

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))


def poisson_blending(src, src_mask, dst, dst_height, dst_width):
    src_height, src_width, _ = src.shape
    dst = dst[0:dst_height, 0:dst_width]
    center = (src_width // 2, src_height // 2)
    blending_image = cv2.seamlessClone(src, dst, src_mask, center, cv2.MONOCHROME_TRANSFER)
    return blending_image


if __name__ == "__main__":
    do_composite()
    for name in tqdm(os.listdir(FG_FOLDER)):
        img = cv2.imread(osp.join(FG_FOLDER, name))
        alpha = cv2.imread(osp.join(A_FOLDER, name))
        if img is None or alpha is None:
            print(name)

