import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def main(args):

    # for user input valid check
    dataset_path = args.dataset
    dataset_path += '/'
    if not os.path.exists(dataset_path):
        raise FileExistsError

    # check the output folder and create if not exists
    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    if not os.path.exists(output_path+'/train/'):
        os.makedirs(output_path+'/'+'train')
        os.makedirs(output_path+'/'+'test')

    # load the data image name
    file_rgb = list()
    for file_name in os.listdir(dataset_path):
        # only process with rgb image data
        if file_name.endswith('_rgb.png'):
            file_rgb.append(file_name.replace('_rgb.png', ''))
    
    # train test split
    train_list, test_list = train_test_split(file_rgb, test_size=args.rate, random_state=0)

    # load the image to corresponding folder
    for file_name in train_list:
        img = cv2.imread(dataset_path+file_name+'_rgb.png')
        cv2.imwrite(output_path+'/'+'train/'+file_name+'_rgb.png', img)

        img = cv2.imread(dataset_path+file_name+'_label.png')
        cv2.imwrite(output_path+'/'+'train/'+file_name+'_label.png', img)
        print('Finish writing:', file_name)

    for file_name in test_list:
        img = cv2.imread(dataset_path+file_name+'_rgb.png')
        cv2.imwrite(output_path+'/'+'test/'+file_name+'_rgb.png', img)

        img = cv2.imread(dataset_path+file_name+'_label.png')
        cv2.imwrite(output_path+'/'+'test/'+file_name+'_label.png', img)
        print('Finish writing:', file_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--format', type=int, required=True, help='Output Image formate: 0 for Gray Scale image, 1 for RGB image', default=0)
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Input dataset path for pre-processing')
    parser.add_argument('-o', '--output_path', type=str, required=False, help='Output dataset path', default='dataset/')
    parser.add_argument('-r', '--rate', type=float, required=False, help='the split rate of the train set and test set', default=0.3)

    args = parser.parse_args()
    main(args)