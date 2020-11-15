from model import UNet
from dataloader import Data_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):

    # load the dataset
    cvppp_dataset = Data_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=cvppp_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    # define RMSprop/Adam
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)


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
    # check cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load picture
    net = UNet(n_channels=1, n_classes=1).to(device=device)
    
    # train
    data_path = "../dataset"
    train_net(net, device, data_path)