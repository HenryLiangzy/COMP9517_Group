import cv2
import os
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def segment(img, mask):
    mask = cv2.bitwise_not(mask)
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    img_out = cv2.subtract(img, img_masked)

    return img_out


def water_shed(img, threshold=50):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, img_thres = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    distance = ndi.distance_transform_edt(img_thres)

    local_maxi = peak_local_max(distance, min_distance=30, indices=False, labels=img_thres)
    markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance, markers, mask=img_thres)

    return labels


def dis_watershed(img, thres_distance=5, threshold=50):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, img_thres = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    distance = ndi.distance_transform_edt(img_thres)

    local_maxi = peak_local_max(distance, min_distance=30, indices=False, labels=img_thres)

    peak_list = list()
    for r in range(local_maxi.shape[0]):
        for c in range(local_maxi.shape[1]):
            if local_maxi[r][c] != False:
                peak_list.append((r, c))

    final_peak = list()
    for r, c in peak_list:
        tag = True
        for peak in final_peak:
            if abs(peak[0]-r)+abs(peak[1]-c) <= thres_distance:
                tag = False
                local_maxi[r][c] = False
                break
        if tag:
            final_peak.append((r, c))

    markers = ndi.label(local_maxi, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance, markers, mask=img_thres)

    return labels


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--dataset', type=str, required=True, help='Input image dataset')
    parse.add_argument('-f', '--footprint_shape', type=int, required=False, help='Footprint shape of WaterShed', default=9)
    parse.add_argument('-t', '--threshold', type=int, required=False, help='WaterShed threshold', default=50)
    args = parse.parse_args()

    # load the prediction image
    image_list = glob.glob(os.path.join(args.dataset, '*_rgb_res.png'))

    threshold = args.threshold
    shape = (args.footprint_shape, args.footprint_shape)

    # start to watershed the image
    for image in image_list:

        image_name = image.replace('_rgb_res.png', '')
        print('Processing', image_name, end='\t')
        
        img_res = cv2.imread(image_name+'_rgb_res.png', 0)
        img_original = cv2.imread(image_name+'_rgb.png')

        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        img_plant = segment(img_original, img_res)

        # two way of segment
        # label = water_shed(img_plant)
        label = dis_watershed(img_plant)

        # save the label image
        plt.imsave(image_name+'_ws.png', label, cmap='gray', format='png')

        print('Done!')
    
