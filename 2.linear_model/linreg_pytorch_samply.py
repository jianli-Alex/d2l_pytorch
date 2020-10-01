#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize linear model by pytorch and its module, such as torch.nn,
torch.utils.data, torch.nn.init, torch.optim, etc.
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.init as init
from sqdm import sqdm


class LinearNet(nn.Module):
    """define a LinearNet"""
    def __init__(self, fea_num):
        super(LinearNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(fea_num, 1)
        )

    def forward(self, x):
        y = self.layer(x)
        return y


"""param"""
params = {
    "input_num": 10000,
    "fea_num": 2,
    "lr": 0.01,
    "batch_size": 128,
    "epoch_num": 10,
}

"""generate data"""
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
x = torch.normal(0, 1, size=(params["input_num"], params["fea_num"]))
error = torch.normal(0, 0.01, size=(params["input_num"], ))
y = torch.mv(x, true_w) + true_b + error
print(y.shape)

"""combine x and y, ild a generator to iter mini-batch data"""
dataset = Data.TensorDataset(x, y)
data_iter = Data.DataLoader(dataset,
                            batch_size=params["batch_size"], shuffle=True)

"""initialize, loss and optimizer"""
net = LinearNet(params["fea_num"])
init.normal_(net.layer[0].weight, 0, 0.01)
# equal to "net.lay[0].bias.data.fill_(0)"
# but fill_ can not use in tensor which in lti-dimension
init.constant_(net.layer[0].bias, 0)
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=params["lr"])

"""training"""
process_bar = sqdm()
for epoch in range(params["epoch_num"]):
    print(f"Epoch [{epoch+1}/{params['epoch_num']}]")
    for xdata, ydata in data_iter:
        # calculate the output of net and loss
        output = net(xdata)
        l = loss(output, ydata.reshape(output.shape))

        # clear grad, equal to "net.zero_grad()"
        optimizer.zero_grad()
        # bp and grad update
        l.backward()
        optimizer.step()

        # train bar
        process_bar.show_process(params["input_num"],
                                 params["batch_size"], round(l.item(), 5))
    print("\n")
print(f"w before update is {true_w}, w after update is {net.layer[0].weight}")
print(f"b before update is {true_b}, b after update is {net.layer[0].bias}")
