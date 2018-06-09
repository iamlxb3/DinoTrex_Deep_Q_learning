import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


class CNN(nn.Module):
    def __init__(self, EPOCH, BATCH_SIZE, LR, fig_wid, fig_len, verbose = False):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # convolution layer (m*n*p)
            nn.Conv2d(              # (1, 28, 28)
                in_channels=1,      # input number of filters
                out_channels=16,    # output number of filters
                kernel_size=5,      # size of filters
                stride=1,           # gap size
                padding=2           # if strid = 1, padding = (kernel_size-1)/2
            ),  # -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential( # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # -> (32, 14, 14)
            nn.ReLU(),                  # -> (32, 14, 14)
            nn.MaxPool2d(2)             # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * int(fig_wid/4) * int(fig_len/4), 1)
        self.fig_wid = fig_wid
        self.fig_len = fig_len
        self.EPOCH = EPOCH
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.is_verbose = verbose


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output

    def data_process(self, train_dataset):
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        return train_loader

    def regressor_train(self, train_dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)  # optimize all cnn parameters
        loss_func = nn.MSELoss()
        train_loader = self.data_process(train_dataset)
        print ("CNN start training...")
        loss_list = []
        for i, epoch in enumerate(range(self.EPOCH)):
            for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when item
                b_x = Variable(x).cuda()  # batch x
                b_y = Variable(y).cuda()  # batch y
                output = self.forward(b_x)  # cnn output

                loss = loss_func(output, torch.unsqueeze(b_y.type(torch.FloatTensor).cuda(), dim=1))  # mean squared error loss
                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()

                if self.is_verbose:
                    if step % 50 == 0:
                        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
            training_loss = loss.data[0]
            loss_list.append(training_loss)
            if i >= 3:
                if loss_list[-1] > loss_list[-2] and loss_list[-1] > loss_list[-3]:
                    print ("NO improvement within 3 epoches! Break!")
                    break
                    s
        print ("CNN training complete...")

    def regressor_dev(self, test_data):
        test_data = Variable(test_data)
        test_output = self.forward(test_data).cpu().data.numpy()[0][0]
        return test_output

    def regressor_train1(self, train_dataset):
        print ("aaaa")