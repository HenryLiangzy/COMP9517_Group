import glob
import numpy as np
import torch
import os
import cv2
from model import UNet
from dataloader import Data_Loader

def main(model_path, dataset_path):
    # get device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model from model file
    # do care load model from ver.GPU or ver.CPU
    net = UNet(n_channels=1, n_classes=1).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    # predice
    net.eval()

    test_loader = Data_Loader(dataset_path)

    # start to predict
    



if __name__ == "__main__":
    main()