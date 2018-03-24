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
import MyResnet as myresnet 

######### Initialize the dataset handler
train_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/ServerTestDictionary_5class.txt")
train_dataset = jdh.Jester_Dataset(dictionary_file=train_dictionary,seq_size=10)

######### Initialize the model
# Custom Resnet50 (the [3,4,6,3] is what makes it Resnet50. Look at orig pytorch resnet.py for other configs)
model = myresnet.ResNet(myresnet.Bottleneck, [3, 4, 6, 3], input_depth=20, num_classes=5)
# Specify the loss function
criterion = nn.CrossEntropyLoss()
# Set the optimization algorithm
optimizer = optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)

# Set batch norms and dropouts in train mode
model.train() 

print("\nTraining...")

######### Training Loop
# Params
b_size = 50
num_epochs = 1
checkpoint_iters = 20
accuracy_report_iters = 2

# Stat keeper
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
        # Print train accuracy every so often so we know its still alive
        if iteration%accuracy_report_iters == 0:
            correct_cnt = 0
            for i in range(b_size):
                # Get the max index from the ith softmax array
                guess = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                # if index == gt label call it correct
                if int(lbl[i]) == int((output.data.max(1, keepdim=True)[1][i])):
                    correct_cnt += 1
            print("Iter: {}, Accuracy: {}".format(iteration,correct_cnt/float(b_size)))

        # Save checkpoint
        if (iteration%checkpoint_iters == 0) and (iteration>0):
            checkpoint_name = 'resnet50_saved_model_{}.pth'.format(iteration)
            print("Checkpointing Model As: ",checkpoint_name)
            torch.save(model, checkpoint_name)

        # Compute the loss
        loss = criterion(output,lbl)
        # Compute the gradients in the backward step
        loss.backward()
        # Run the optimizer and update the weights
        optimizer.step()
        scheduler.step()
        iteration += 1

# Saving Model
print("Saving Model...")
torch.save(model, './resnet50_saved_model_FINAL.pth')


