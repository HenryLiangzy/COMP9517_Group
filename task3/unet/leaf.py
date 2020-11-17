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


def water_shed(img, threshold=50, footprint_shape=(9, 9)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    ret, img_thres = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    distance = ndi.distance_transform_edt(img_thres)

    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones(footprint_shape), labels=img_thres)
    markers = ndi.label(local_maxi)[0]
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
        label = water_shed(img_plant, threshold=threshold, footprint_shape=shape)

        # save the label image
        plt.imsave(image_name+'_ws.png', label, cmap=plt.cm.nipy_spectral, format='png')

        print('Done!')
    
