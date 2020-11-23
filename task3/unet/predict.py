import glob
import numpy as np
import torch
import os
import cv2
import argparse
from model import UNet
from dataloader import Data_Loader

def predict(model_path, dataset_path, predict_type):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1).to(device)
    
    # load model from model path
    net.load_state_dict(torch.load(model_path, map_location=device))

    # evaluation model
    net.eval()
    # load test image path
    tests_path = glob.glob(os.path.join(dataset_path, '*_rgb.png'))

    for test_path in tests_path:
        print('Processing', test_path)

        save_res_path = test_path.split('.')[0] + '_res.png'

        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # reshape the size of input image
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        
        # convert to tensor and load to device
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        
        # predict the image
        pred = net(img_tensor)

        # de-normalization 
        pred = np.array(pred.data.cpu()[0])[0] * 255
        # ret, pred = cv2.threshold(pred, 1, 255, cv2.THRESH_BINARY)

        # save file
        cv2.imwrite(save_res_path, pred)
    

if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('-d', '--dataset', type=str, required=True, help='Input predict dataset')
    parse.add_argument('-m', '--model_path', type=str, required=True, help='model path')
    parse.add_argument('-t', '--predict_type', type=int, required=False, help='Predict type, 0 for gray, 1 for scale', default=0)
    args = parse.parse_args()

    predict(args.model_path, args.dataset, args.predict_type)