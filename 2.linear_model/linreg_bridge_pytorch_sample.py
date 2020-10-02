#! /usr/bin/env python
# -*-coding: utf-8 -*-

"""
function: realize bridge linear model by pytorch with weight decay
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as Data
from sqdm import sqdm
import warnings
warnings.filterwarnings("ignore")


class PLinearBridge2(nn.Module):
    def __init__(self, fea_num):
        super(PLinearBridge2, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(fea_num, 1)
        )

    def forward(self, x):
        y = self.layer(x)
        return y


params = {
    "input_num": 10000,
    "fea_num": 2,
    "batch_size": 128,
    "lr": 0.01,
    "weight_decay": 0.05,
    "epoch_num": 10,
}

"""generate data"""
true_w = torch.tensor([2, -3.4])
true_b = torch.tensor([4.2])
torch.manual_seed(params["input_num"])
x = torch.normal(0, 1, size=(params["input_num"], len(true_w)))
error = torch.normal(0, 0.01, size=(params["input_num"],))
y = torch.mv(x, true_w) + true_b + error
# combine x and y
dataset = Data.TensorDataset(x, y)
data_iter = Data.DataLoader(dataset, params["batch_size"], shuffle=True)

# generate network and init
net = PLinearBridge2(params["fea_num"])
torch.manual_seed(100)
init.normal_(net.layer[0].weight, 0, 0.01)
init.constant_(net.layer[0].bias, 0)
# loss and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=params["lr"],
                            weight_decay=params["weight_decay"])

# training bar
process_bar = sqdm()
for epoch in range(params["epoch_num"]):
    print(f"Epoch [{epoch}/{params['epoch_num']}]")
    for xdata, ydata in data_iter:
        y_pred = net(xdata)
        l = loss(y_pred, ydata.reshape(y_pred.shape))
        # clear grad and bp
        optimizer.zero_grad()
        l.backward()
        # update
        optimizer.step()
        process_bar.show_process(params["input_num"],
                                 params["batch_size"], round(l.item(), 5))
    print("\n")
print(f"w before update is {true_w}, w after update is {net.layer[0].weight}")
print(f"b before update is {true_b}, b after update is {net.layer[0].bias}")