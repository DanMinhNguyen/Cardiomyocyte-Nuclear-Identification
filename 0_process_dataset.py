import math
import os
import random
from multiprocessing import Pool

import gc
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread, imwrite
from skimage.filters import threshold_otsu, threshold_local
from skimage.morphology import opening, binary_dilation


def optimize_k(A, B, iter=100, evalcost=False):
    sigstep = np.divide(1, np.abs(A), out=np.zeros_like(A, dtype=float), where=A != 0)
    taustep = 1 / np.sum(np.abs(A))
    xp = 0
    xn = 0
    z = np.zeros_like(A, dtype=float)
    theta = 1
    cost = []
    klist = []

    for _ in range(iter):
        #         print(_)
        xp = xn - taustep * np.sum(np.multiply(A, z))
        xhat = xp + theta * (xp - xn)
        z = np.minimum(1, np.maximum(-1, z + np.multiply(sigstep, A * xhat) - np.multiply(sigstep, B)))

        xn = xp

        if evalcost:
            cost.append(np.sum(np.abs(A * xp - B)))
            klist.append(xp)
    return xp, cost, klist

def process_slide(filename, IMG_SIZE=256, E_skip=8, filter=True, debug_imgs=False,
                  save_imgs=True):

    data = imread("raw_data" + filename).astype(np.float32)

    for ii in range(data.shape[-1]):
        data[..., ii] = data[..., ii] / data[..., ii].max()

    data = np.flip(data, axis=2)

    print(filename)
    print('\toptimizing k...')

    xcen = int(data.shape[0] / 2)
    ycen = int(data.shape[1] / 2)
    s = 1024
    k, _, _ = optimize_k(data[xcen-s:xcen+s,ycen-s:ycen+s,1],data[xcen-s:xcen+s,ycen-s:ycen+s,2])

    print('\tfound k as ',str(k))

    buff = data[..., 2] - k * data[..., 1]
    buff[buff<0] = 0
    data[..., 2] = buff

    nucthresh = threshold_otsu(data[..., 0])
    myothresh = threshold_otsu(data[..., 2])

    print(filename + " shape " + str(data.shape))

    img_shape = data.shape
    tile_size = (IMG_SIZE, IMG_SIZE)
    offset = (IMG_SIZE, IMG_SIZE)

    X_list = [data[offset[1] * i:min(offset[1] * i + tile_size[1], img_shape[0]),
              offset[0] * j:min(offset[0] * j + tile_size[0], img_shape[1])]
              for i in range(E_skip, int(math.floor(data.shape[0] / (offset[1] * 1.0))) - E_skip)
              for j in range(E_skip, int(math.floor(data.shape[1] / (offset[0] * 1.0))) - E_skip)]

    if filter:
        X_list = [crop for crop in X_list if np.amax(crop[..., 0]) > .1 and np.mean(crop[..., 0]) < .45
                  and .05 < np.mean(crop[..., 1]) < .8 and np.mean(crop[..., 2]) < .5]

    X_crops = np.asarray(X_list, dtype=np.float32)

    X_crops_copy = np.copy(X_crops[..., 2])

    print("cropped shape " + str(X_crops.shape))

    nuclei_masks = np.empty_like(X_crops_copy)
    for i in range(X_crops.shape[0]):
        mask = X_crops[i, ..., 0] > nucthresh
        opening(mask, out=mask)
        nuclei_masks[i, ...] = mask

    for i in range(X_crops.shape[0]):
        X_crops[i, ..., 2] = np.clip(X_crops[i, ..., 2] - np.mean(X_crops[i, ..., 2]), 0, 1)
        if np.sum(X_crops[i, ..., 2], axis=(0, 1)) != 0:
            mask = X_crops[i, ..., 2] > myothresh
            opening(mask, out=mask)
            X_crops[i, ..., 2] = mask
            X_crops[i, ..., 2] = X_crops[i, ..., 2] * nuclei_masks[i, ...]

    if debug_imgs:
        f, axarr = plt.subplots(4, 5, figsize=(20, 8))
        for i in range(5):
            ix = random.randint(0, X_crops.shape[0] - 1)

            im = axarr[0, i].imshow(X_crops[ix, ..., 0])
            plt.colorbar(im, ax=axarr[0, i])
            axarr[0, i].title.set_text('Channel 0')

            im = axarr[1, i].imshow(nuclei_masks[ix, ...])
            plt.colorbar(im, ax=axarr[1, i])
            axarr[1, i].title.set_text('Channel 1')

            im = axarr[2, i].imshow(X_crops[ix, ..., 2])
            plt.colorbar(im, ax=axarr[2, i])
            axarr[2, i].title.set_text('Channel 2')

            im = axarr[3, i].imshow(np.clip(X_crops_copy[ix, ...] - np.mean(X_crops_copy[ix, ...]), 0, 1))
            plt.colorbar(im, ax=axarr[3, i])
            axarr[3, i].title.set_text('Channel 3')
        plt.show()

    count = 0
    if save_imgs:
        for i in range(X_crops.shape[0]):
            if .0008 < np.mean(X_crops[i, ..., 2]) < .15:
                imwrite("train_data/{filename}_{i:0>4d}.tiff", X_crops[i, ...])
                count += 1

    print("processing complete! imgs saved: " + str(count))
    gc.collect()


if __name__ == "__main__":
    slide_locs = next(os.walk("raw_data"))[2]
    print(slide_locs)
    for slide in slide_locs:
        process_slide(slide, IMG_SIZE=256, E_skip=0, filter=True, debug_imgs=False, save_imgs=True)
