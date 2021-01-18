import numpy as np
from torchvision import transforms
from tqdm import tqdm
import pickle as pkl
import torch
import pickle
import time
import torchvision
from torch.utils.data import DataLoader
from model import *
from layers import AdamGD, SGD
from torchvision import datasets, transforms
import torch
import setting

device = setting.device


def evalHot(y, pred):
    """
    评估效果
    :param y:真实值的独热编码
    :param pred: 预测值的输出
    :return: 正确的个数
    """
    _y = torch.argmax(y, dim=-1).cpu().numpy()
    _pred = torch.argmax(pred, dim=-1).cpu().numpy()
    N = np.sum((_y == _pred))
    return N


def LoadMNIST(root, transform, batch_size, download=True):
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def OneHotLabel(Y, n):
    """
    :param Y:序列型标签
    :param n: 标签数目
    :return: 标签的独热编码
    """
    y = torch.zeros([len(Y), n],device=device)
    y[torch.arange(0, len(Y), device=device), Y] = 1
    return y


def CELoss(Y, T):
    """
    :param Y:模型输出
    :param T: 样本标签
    :return: 交叉熵损失
    """
    return -(T * torch.log(Y)).sum(dim=-1)


def KMeansRepeatY(Y, repeat):
    # print(Y.shape)
    repeatY = torch.cat([Y] * repeat, dim=0)
    return repeatY


def KMeansRepeatX(X, repeat):
    """
    :param X:Raw data \\in R^{batch_size X n_dim}
    :param repeat:重复的次数、采样数
    :return: 加了偏置项和重复数据的样本 维度[batch_size,repeat,n_dum+1]
    """
    X = X.reshape(len(X), -1)
    repeatX = torch.cat([X] * repeat, dim=0)
    return repeatX


if __name__ == "__main__":
    train_loss = []
    test_loss = []
    acc = []
    time_list = []
    epoch_train_estimation_relative_error = []
    epoch_test_estimation_relative_error = []
    repeat_n = 100
    alg_start = 0.
    alg_end = 0.
    learning_rate = 1e-1

    batch_size = 128
    epoches = 20
    loss = 0.

    start_epoch = 0
    tranform = transforms.Compose([transforms.ToTensor()])

    import os

    if not os.path.exists('./mnist_model'):
        os.mkdir('mnist_model')
    if not os.path.exists('./mnist_logs'):
        os.mkdir('mnist_logs')
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=0)
    # test_dataloader = DataLoader(dataset=val_dataset, batch_size=128, num_workers=0, shuffle=False)
    train_dataloader, test_dataloader = LoadMNIST('../../LR/data/MNIST', tranform, batch_size, False)
    # net = LeNet5()
    net = ShallowConvNet()
    # train_img, train_label = loadMNIST_RAM(train_dataloader, repeat_n)
    print('数据准备完成!')
    trainLoss = 0.
    testLoss = 0.
    print('epoch to run:{} learning rate:{}'.format(epoches, learning_rate))
    start = time.time()
    alg_start = time.time()
    optimizer = SGD(lr=learning_rate, params=net.get_params())
    for epoch in range(start_epoch, start_epoch + epoches):
        start = time.time()
        loss = 0.
        nbatch = 0.
        N = 0.
        n = 0.
        trainLoss = 0.
        train_estimation_relative_error = 0
        for batch, [trainX, trainY] in enumerate(tqdm(train_dataloader, ncols=10)):
            # break
            trainX = KMeansRepeatX(trainX, repeat_n).reshape(-1, 1, 28, 28).to(device)
            trainY = OneHotLabel(trainY, 10).to(device)
            trainY = KMeansRepeatY(trainY, repeat_n).to(device)

            nbatch += 1

            pre = net.forward(trainX)

            loss = CELoss(pre, trainY).cpu().numpy()
            trainLoss += np.mean(loss)

            grads = net.backward(pre, trainY)
            params = optimizer.update_params(grads)
            net.set_params(params)

        trainLoss /= nbatch
        train_loss.append(trainLoss)
        epoch_train_estimation_relative_error.append(train_estimation_relative_error / nbatch)
        # trainAcc = n / N
        print('train epoch:{} loss:{}'.format(epoch, trainLoss))
        if ((epoch + 1) % 10 == 0):
            learning_rate *= 0.8
            print('学习率衰减至{}'.format(learning_rate))
        loss = 0.
        N = 0.
        n = 0.
        nbatch = 0.
        test_estimation_relative_error = 0
        for batch, [testX, testY] in enumerate(tqdm(test_dataloader, ncols=10)):
            nbatch += 1
            testX = testX.reshape(-1, 1, 28, 28).to(device)
            testY = OneHotLabel(testY, 10).to(device)
            pre = net.forward(testX)
            testLoss += np.mean(CELoss(pre, testY).cpu().numpy())
            N += len(testX)
            n += evalHot(testY, pre)
        testLoss /= nbatch
        test_loss.append(testLoss)
        testAcc = n / N
        acc.append(testAcc)
        epoch_test_estimation_relative_error.append(test_estimation_relative_error / nbatch)
        print('test epoch:{} loss:{} acc:{}'.format(epoch, testLoss, n / N))
        time_list.append(time.time() - start)
