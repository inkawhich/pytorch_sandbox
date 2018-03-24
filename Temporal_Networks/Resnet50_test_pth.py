# NAI

# This script will test a saved resnet50 checkpoint which was trained in Resnet50_train_and_save.py.

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
import MyResnet as myresnet 

# Initialize the dataset handler
test_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/ServerTestDictionary_5class.txt")
test_dataset = jdh.Jester_Dataset(dictionary_file=test_dictionary,seq_size=10)

# Initialize the model
# Custom Resnet50 (the [3,4,6,3] is what makes it Resnet50. Look at orig pytorch resnet.py for other configs)
model = myresnet.ResNet(myresnet.Bottleneck, [3, 4, 6, 3], input_depth=20, num_classes=5)
# Load the model from the checkpoint
model = torch.load('./resnet50_saved_model_20.pth')
# Set it to test mode (for batch norm and dropout layers)
model.eval()

# Testing loop
correct_cnt = 0
total_cnt = 0

for img,lbl,seq in test_dataset.read(batch_size=1, shuffle=True):
    # Create the data and label variables so we can use them in the computation
    img = Variable(torch.FloatTensor(img),requires_grad=False)
    lbl = Variable(torch.LongTensor(lbl))
    # Call a forward pass on the data
    output = model(img)
    # Run quick accuracy check
    #print("GT Label: ",lbl.data.numpy())
    #print("Output: ",output.data.numpy())
    # Get the max index from the ith softmax array
    #guess = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    # if index == gt label call it correct
    if int(lbl[0]) == int((output.data.max(1, keepdim=True)[1][0])):
        correct_cnt += 1
    total_cnt += 1
    if total_cnt%100 == 0:
        print("Accuracy@{}: {}".format(total_cnt,correct_cnt/float(total_cnt)))
print("\nFINAL Accuracy: {}".format(correct_cnt/float(total_cnt)))


