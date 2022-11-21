import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def buildAndTrainTheModel(x, y):
    # build model
    ANNreg = nn.Sequential(
        nn.Linear(1, 1),  # input layer
        nn.ReLU(),  # activation function
        nn.Linear(1, 1)  # output layer
    )

    # loss function
    lossfun = nn.MSELoss()
    # optimizer (the flavor of gradient descent to implement
    optimizer = torch.optim.SGD(ANNreg.parameters(), lr=.05)
    # train the model
    numepochs = 500
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

    return predictions, losses


def createTheDate(m):
    N = 50
    x = torch.randn(N, 1)
    y = m * x + torch.randn(N, 1) / 2
    plt.plot(x, y, 's')

    return x, y


def testAnnModel():
    x, y = createTheDate(.8)

    yHat, losses = buildAndTrainTheModel(x, y)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses.detach(), 'o', markerfacecolor='w', linewidth=0.1)
    ax[0].set_xlabel('epoch')
    ax[0].set_title('Loss')

    ax[1].plot(x, y, 'bo', label='real data')
    ax[1].plot(x, yHat.detach(), 'rs', label='predictions')
    ax[1].set_xlabel('x')
    ax[1].set_label('y')
    ax[1].set_title(f'predition-data corrr ={np.corrcoef(y.T, yHat.detach().T)[0, 1]:.2f}')
    ax[1].legend()
    plt.show()


slopes = np.linspace(-2, 2, 21)
numExps = 50
results = np.zeros((len(slopes), numExps, 2))
for slopei in range(len(slopes)):

    for N in range(numExps):
        x, y = createTheDate(slopes[slopei])
        yHat, losses = buildAndTrainTheModel(x, y)

        results[slopei, N, 0] = losses[-1]
        results[slopei, N, 1] = np.corrcoef(y.T, yHat.detach().T)[0, 1]

results[np.isnan(results)] = 0

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(slopes, np.mean(results[:, :, 0], axis=1), 'ko-', markerfacecolor='w', markersize=10)
ax[0].set_xlabel('slope')
ax[0].set_title('Loss')

ax[1].plot(slopes, np.mean(results[:, :, 1], axis=1), 'ms-', markerfacecolor='w', markersize=10)
ax[1].set_xlabel('slope')
ax[1].set_label('real_predicted correlation')
ax[1].set_title('model performance')

plt.show()
