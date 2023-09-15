from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2


def preprocessor(img, imgSize, enhance=False, dataAugmentation=False):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])
        print("Image None!")
    if dataAugmentation:
        stretch = (random.random() - 0.5)
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
        img = cv2.resize(img, (wStretched, img.shape[0]))

    if enhance:
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(imgContrast, kernel, iterations=1)

    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img
    img = cv2.transpose(target)

    return img


def wer(r, h):
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)]
