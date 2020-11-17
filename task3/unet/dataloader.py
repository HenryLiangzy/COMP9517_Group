import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

# You should build your custom dataset as below.
class Data_Loader(torch.utils.data.Dataset):
    def __init__(self, dataset_path):
        '''Initialize file paths of image dataset'''

        self.image_path = glob.glob(os.path.join(dataset_path, '*_rgb.png'))  

    def __getitem__(self, index):
        '''
        1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        2. Preprocess the data (e.g. torchvision.Transform).
        3. Return a data pair (e.g. image and label).
        '''

        # extrat the image name for further processing
        img_name = self.image_path[index].replace('_rgb.png', '')

        # load the image data from the dataset in gray scale
        rgb_img = cv2.imread(img_name+'_rgb.png')
        label_img = cv2.imread(img_name+'_label.png')
        
        # convert to single channel
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

        rgb_img = rgb_img.reshape(1, rgb_img.shape[0], rgb_img.shape[1])
        label_img = label_img.reshape(1, label_img.shape[0], label_img.shape[1])

        if label_img.max() > 1:
            label_img = label_img / 255

        return rgb_img, label_img


    def __len__(self):
        ''' Return the len of the data set '''
        return len(self.image_path)


# Follow is for class testing
if __name__ == "__main__":
    cvppp_dataset = Data_Loader('../dataset/train')
    print('dataset size:', len(cvppp_dataset))

    train_loader = torch.utils.data.DataLoader(dataset=cvppp_dataset, batch_size=1, shuffle=True)
    for img, label in train_loader:
        print(img.shape)