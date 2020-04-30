import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.filters import threshold_otsu
from skimage.morphology import opening
from scipy.ndimage import label
from tqdm import tqdm

def read_image_data(datafile):
    im = imread(datafile).astype(np.float32)
    for ii in range(im.shape[-1]):
        im[..., ii] = im[..., ii] / 255
    return im


def binary_object_extraction(labeled_image=None, num_objects=None):
    separated_image = np.zeros(((num_objects,) + labeled_image.shape), dtype=np.bool)
    for ii in tqdm(range(num_objects)):
        separated_image[ii][labeled_image == (ii + 1)] = 1
    return separated_image


def get_nuclei_masks(data, version='semantic'):
    nucthresh = threshold_otsu(data[..., 2])
    mask = data[..., 2] > nucthresh
    opening(mask, out=mask)
    if version == 'semantic':
        return mask
    elif version == 'instance':
        rlabel, rnum = label(mask)
        sepmask = binary_object_extraction(labeled_image=rlabel, num_objects=rnum)
        return sepmask
    else:
        print('incorrect version specification')
        print('please use "semantic" or "instance"')


def get_mc_masks(data, k):
    mcdata = data[..., 0] - k * data[..., 1]
    mcdata[mcdata < 0] = 0
    myothresh = threshold_otsu(mcdata)
    mcmask = mcdata > myothresh
    opening(mcmask, out=mcmask)
    opening(mcmask, out=mcmask)
    mcmask = np.multiply(mcmask, get_nuclei_masks(data, 'semantic'))
    return mcmask


def optimize_k(data, iter=100, evalcost=False):
    A = data[..., 1]
    B = data[..., 0]
    sigstep = np.divide(1, np.abs(A), out=np.zeros_like(A, dtype=float), where=A != 0)
    taustep = 1 / np.sum(np.abs(A))
    xp = 0
    xn = 0
    z = np.zeros_like(A, dtype=float)
    theta = 1
    cost = []
    klist = []

    for _ in tqdm(range(iter)):
        #         print(_)
        xp = xn - taustep * np.sum(np.multiply(A, z))
        xhat = xp + theta * (xp - xn)
        z = np.minimum(1, np.maximum(-1, z + np.multiply(sigstep, A * xhat) - np.multiply(sigstep, B)))

        xn = xp

        if evalcost:
            cost.append(np.sum(np.abs(A * xp - B)))
            klist.append(xp)

    if evalcost:
        plt.figure(figsize=(16, 8))
        plt.plot(cost)
        plt.title('cost')
        plt.xlabel('iterations')
        plt.show()

        plt.figure(figsize=(16, 8))
        plt.plot(klist)
        plt.title('k')
        plt.xlabel('iterations')
        plt.show()

    return xp


def patchwise_predict2D(x, model, numsize=(256, 256), stride=(128, 128)):
    X, Y = np.meshgrid(np.linspace(-1, 1, numsize[1]), np.linspace(-1, 1, numsize[0]))
    mu, sigma = 0, 2.5
    G = np.exp(-((X - mu) ** 2 + (Y - mu) ** 2) / 2.0 * sigma ** 2)

    ns_row = np.ceil((x.shape[0] - numsize[0]) / stride[0]).astype(int)
    ns_col = np.ceil((x.shape[1] - numsize[1]) / stride[1]).astype(int)

    padrow = ns_row * stride[0] + numsize[0] - x.shape[0]
    padcol = ns_col * stride[1] + numsize[1] - x.shape[1]

    npad = ((padrow, 0), (padcol, 0), (0, 0))

    x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    ypred = np.zeros((x.shape[0], x.shape[1]))
    yweight = np.zeros((x.shape[0], x.shape[1]))

    progress = 0
    tot_progress = (ns_row + 1) * (ns_col + 1)
    for i in tqdm(range(ns_row + 1)):
        for j in range(ns_col + 1):
            row = i * stride[0]
            col = j * stride[1]
            xbuff = np.zeros((1, numsize[0], numsize[1], x.shape[-1]))
            xbuff[0, :, :, :] = x[row:row + numsize[0], col:col + numsize[1], :]
            ybuff = model.predict(xbuff)

            ypred[row:row + numsize[0], col:col + numsize[1]] = (
                np.divide(np.multiply(ypred[row:row + numsize[0], col:col + numsize[1]],
                                      yweight[row:row + numsize[0], col:col + numsize[1]])
                          + np.multiply(ybuff[0, :, :, 0], G),
                          G + yweight[row:row + numsize[0], col:col + numsize[1]]))
            yweight[row:row + numsize[0], col:col + numsize[1]] += G
            progress += 1
            # print(progress/tot_progress*100," percent complete         \r",)
    print("")
    ypred = np.delete(ypred, range(padrow), 0)
    ypred = np.delete(ypred, range(padcol), 1)
    return ypred


def predict_mc(data, model):
    x = np.flip(data, axis=2)[..., 0:2]
    ypred = patchwise_predict2D(x, model, numsize=(256, 256), stride=(128, 128))
    return ypred


def patchwise_mc_nuclei_masks(x, nuclei='mc', numsize=(256, 256), stride=(256, 256)):
    X, Y = np.meshgrid(np.linspace(-1, 1, numsize[1]), np.linspace(-1, 1, numsize[0]))
    mu, sigma = 0, 2.5
    G = np.exp(-((X - mu) ** 2 + (Y - mu) ** 2) / 2.0 * sigma ** 2)

    ns_row = np.ceil((x.shape[0] - numsize[0]) / stride[0]).astype(int)
    ns_col = np.ceil((x.shape[1] - numsize[1]) / stride[1]).astype(int)

    padrow = ns_row * stride[0] + numsize[0] - x.shape[0]
    padcol = ns_col * stride[1] + numsize[1] - x.shape[1]

    npad = ((padrow, 0), (padcol, 0), (0, 0))

    x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    ypred = np.zeros((x.shape[0], x.shape[1]))
    yweight = np.zeros((x.shape[0], x.shape[1]))

    progress = 0
    tot_progress = (ns_row + 1) * (ns_col + 1)
    for i in tqdm(range(ns_row + 1)):
        for j in range(ns_col + 1):
            row = i * stride[0]
            col = j * stride[1]
            xbuff = np.zeros((numsize[0], numsize[1], x.shape[-1]))
            xbuff[...] = x[row:row + numsize[0], col:col + numsize[1], :]

            if nuclei == 'mc':
                k = optimize_k(xbuff)
                ybuff = get_mc_masks(xbuff, k)
            elif nuclei == 'all':
                ybuff = get_nuclei_masks(xbuff, version='semantic')

            ypred[row:row + numsize[0], col:col + numsize[1]] = (
                np.divide(np.multiply(ypred[row:row + numsize[0], col:col + numsize[1]],
                                      yweight[row:row + numsize[0], col:col + numsize[1]])
                          + np.multiply(ybuff[:, :], G),
                          G + yweight[row:row + numsize[0], col:col + numsize[1]]))
            yweight[row:row + numsize[0], col:col + numsize[1]] += G
            progress += 1
            # print(progress/tot_progress*100," percent complete         \r",)
    print("")
    ypred = np.delete(ypred, range(padrow), 0)
    ypred = np.delete(ypred, range(padcol), 1)

    ypred = np.round(ypred).astype(bool)
    return ypred


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def analyze_data_onthefly(data, tmask, pmask, nmask, thresholdlist=[0]):
    tp = np.zeros(shape=(len(thresholdlist),))
    tn = np.zeros(shape=(len(thresholdlist),))
    fp = np.zeros(shape=(len(thresholdlist),))
    fn = np.zeros(shape=(len(thresholdlist),))

    if nmask is None:
        nmask = get_nuclei_masks(data, version='semantic')
        rlabel, rnum = label(nmask)
    else:
        rlabel, rnum = label(nmask)

    for ii in tqdm(range(rnum)):
        rmask = (rlabel == (ii + 1))
        rmin, rmax, cmin, cmax = bbox(rmask)
        tval = np.sum(np.multiply(rmask[rmin:rmax, cmin:cmax], tmask[rmin:rmax, cmin:cmax]))
        pval = np.sum(np.multiply(rmask[rmin:rmax, cmin:cmax], pmask[rmin:rmax, cmin:cmax])) / np.count_nonzero(
            rmask[rmin:rmax, cmin:cmax])

        for jj in range(len(thresholdlist)):
            threshold = thresholdlist[jj]
            if (pval > threshold):
                if (tval > 0):
                    tp[jj] += 1
                else:
                    fp[jj] += 1
            else:
                if (tval > 0):
                    fn[jj] += 1
                else:
                    tn[jj] += 1

    return tp, tn, fp, fn, np.array(threshold)


def calc_sens_spec(tp, tn, fp, fn):
    # sensitivity, recall, hit rate, true positve rate (TPR), 1-FNR
    sensitivity = np.divide(tp, tp + fn)
    # specificity, selectivity or true negative rate (TNR), 1-FPR
    specificity = np.divide(tn, tn + fp)
    return sensitivity, specificity