import torch
import numpy as np
import torch.nn as nn
import torchvision
import ipdb
import torchvision.transforms as transforms

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def train_model(self, data_loader, epoch, step_size, save_chkpnt=True):
        lowest_epoch_loss = float('inf')
        for n in range(epoch):
            losses = 0
            for i in range(step_size):
                x_batch, y_batch = next(data_loader)
                x_batch = torch.autograd.Variable(x_batch).cuda()
                y_batch = torch.from_numpy(y_batch).float().cuda()
                # Forward pass
                outputs = self.forward(x_batch)
                loss = self.loss_func(outputs, y_batch)
                losses += loss
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.data[0]))
            epoch_loss = losses / n
            if save_chkpnt:
                if epoch_loss < lowest_epoch_loss:
                    lowest_epoch_loss = epoch_loss
                    torch.save(self, 'cnnchkpnt_loss_{}_epoch_{}'.format(epoch_loss, epoch))


if __name__ == '__main__':
    image_size = (100, 1, 28, 28)
    net = ConvNet(10)
    print(net)

    # Hyper parameters
    num_epochs = 5
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)


    model = ConvNet(num_classes)

    if torch.cuda.is_available():
        model = model.cuda()
        print("Cuda is available!")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = torch.autograd.Variable(images).cuda()
            labels = torch.autograd.Variable(labels).cuda()

            labels = np.ones((100,10))
            labels = torch.from_numpy(labels).float().cuda() # torch.unsqueeze(b_y.type(torch.FloatTensor).cuda()


            # Forward pass
            outputs = model(images)
            #loss = criterion(outputs, labels)
            loss = loss_func(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.data[0]))
            # import ipdb
            # ipdb.set_trace()
