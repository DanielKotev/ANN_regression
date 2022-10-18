import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

N = 30
x = torch.randn(N, 1)
y = x + torch.randn(N, 1) / 2

plt.plot(x, y, 's')
plt.show()

# build model
ANNreg = nn.Sequential(
    nn.Linear(1, 1),  # input layer
    nn.ReLU(),  # activation function
    nn.Linear(1, 1)  # output layer
)

# learning rate
learningRate = .05
# loss function
lossfun = nn.MSELoss()
# optimizer (the flavor of gradient descent to implement
optimizer = torch.optim.SGD(ANNreg.parameters(), lr=learningRate)
# train the model
numepochs = 250
losses = torch.zeros(numepochs)

# Train the model 2
for epochi in range(numepochs):
    # forward pass
    yHat = ANNreg(x)

    # compute the loss
    loss = lossfun(yHat, y)
    losses[epochi] = loss

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# show the losses
predictions = ANNreg(x)

testloss = (predictions - y).pow(2).mean()
plt.plot(losses.detach(), 'o', markerfacecolor='w', linewidth=.1)
plt.plot(numepochs,testloss.detach(),'ro')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Final loss = %g'%testloss.item())
plt.show()