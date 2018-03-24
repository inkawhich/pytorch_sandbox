# NAI

# This script will try to adapt the pytorch implementation of resnet to use optical flow
#  stacks. The model code is from here: 
#
#     https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

from __future__ import print_function
from __future__ import division

import os
import numpy as np
import struct
import random
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import JesterDatasetHandler as jdh

##############################################################################
# Define the model stuff 
# (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
##############################################################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_depth=3, last_kernel=7, num_classes=5):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_depth, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(last_kernel, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Had to add this to avoid size mismatches
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Initialize the dataset handler
train_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/ServerTestDictionary_5class.txt")
train_dataset = jdh.Jester_Dataset(dictionary_file=train_dictionary,seq_size=10)

# Initialize the model
#model = models.resnet50()
# Custom Resnet50
model = ResNet(Bottleneck, [3, 4, 6, 3], input_depth=20, last_kernel=3, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=5e-4)

################################
# Run Training
################################
print("\nTraining...")

b_size = 50
num_epochs = 1

# Training Loop
model.train() # I think this sets the dropouts in train mode

iteration = 0
for e in range(num_epochs):
    for img,lbl,seq in train_dataset.read(batch_size=b_size, shuffle=True):
        # Create the data and label variables so we can use them in the computation
        img = Variable(torch.FloatTensor(img))
        lbl = Variable(torch.LongTensor(lbl))
        # Zero out whatever the gradients are currently
        optimizer.zero_grad() # make sure the gradient members are all zero before running
        # Call a forward pass on the data
        output = model(img)
        # Run quick accuracy check
        #print("GT Label: ",lbl)
        #print("Output: ",output)
        if iteration%2 == 0:
            correct_cnt = 0
            for i in range(b_size):
                # Get the max index from the ith softmax array
                guess = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                # if index == gt label call it correct
                if int(lbl[i]) == int((output.data.max(1, keepdim=True)[1][i])):
                    correct_cnt += 1
            print("Iter: {}, Accuracy: {}".format(iteration,correct_cnt/float(b_size)))

        # Compute the loss
        #loss = F.nll_loss(output,lbl)
        loss = criterion(output,lbl)
        # Compute the gradients in the backward step
        loss.backward()
        # Run the optimizer and update the weights
        optimizer.step()
        iteration += 1

# Saving Model
print("Saving Model...")
torch.save(model, './resnet50_saved_model.pth')


