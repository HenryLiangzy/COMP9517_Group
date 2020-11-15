from model import UNet
from dataloader import Data_Loader
from torch import optim
import torch.nn as nn
import torch
import argparse

def train_net(net, device, data_path, epochs, batch_size, lr):

    # load the dataset
    cvppp_dataset = Data_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=cvppp_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    # define RMSprop/Adam
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.toptim.Adam(net.parameters(), lr=0.01)

    # define Loss function
    criterion = nn.BCEWithLogitsLoss()

    # init best_loss
    best_loss = float('inf')

    # Loop epoch time
    for epoch in range(epochs):

        net.train()
        # train with batch size
        for image, label in train_loader:
            optimizer.zero_grad()

            # copy data to device
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # Predict and calculate the loss
            pred = net(image)
            loss = criterion(pred, label)


            # backpropagation parameter, Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimizer.step()

        # show loss after each epoch
        print('Epoch', epoch, 'Loss:', loss.item())

        # Save the smallest loss model
        if loss < best_loss:
            best_loss = loss
            torch.save(net.state_dict(), 'best_model.pth')


# Train here
if __name__ == "__main__":
    
    # user figure
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Input dataset path')
    parser.add_argument('-e', '--epoch', type=int, required=False, help='Epoch time', default=10)
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='Batch size of training set', default=2)
    parser.add_argument('-l', '--lr', type=float, required=False, help='Lr value of model', default=0.00001)
    args = parser.parse_args()

    # check cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load picture
    net = UNet(n_channels=1, n_classes=1).to(device=device)
    
    # train
    data_path = args.dataset
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    train_net(net, device, data_path, epoch, batch_size, lr)