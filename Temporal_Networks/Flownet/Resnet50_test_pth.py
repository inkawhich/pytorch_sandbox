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
import JesterDatasetHandler2 as jdh   # Diff for 5/10 class
import MyResnet as myresnet 


# 1=GPU_Mode; 0=CPU_Mode
GPU = 1


# Initialize the dataset handler
# 5 class
#test_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/TestDictionary_5class.txt")
# 10 class
test_dictionary = os.path.join(os.path.expanduser('~'),"DukeML/datasets/jester/alt_TestDictionary_10class.txt")

test_dataset = jdh.Jester_Dataset(dictionary_file=test_dictionary,seq_size=10)

# Initialize the model
# Custom Resnet50 (the [3,4,6,3] is what makes it Resnet50. Look at orig pytorch resnet.py for other configs)
model = myresnet.ResNet(myresnet.Bottleneck, [3, 4, 6, 3], input_depth=20, num_classes=10)

# For GPU support
if GPU:
    import torch.nn.parallel
    import torch.backends.cudnn as cudnn
    import torch.utils.data
    model = torch.nn.DataParallel(model).cuda()

# Load the model from the checkpoint
# Note alt10 best @ 110k ~ 86%
#model = torch.load('./resnet50_alt_10class_saved_model_110000.pth')
model = torch.load('./resnet50_ft_farneback_alt_10class_saved_model_115000.pth')
# Set it to test mode (for batch norm and dropout layers)
model.eval()

# Testing loop
correct_cnt = 0
total_cnt = 0
cmat = np.zeros((10,10))

for img,lbl,seq in test_dataset.read(batch_size=1, shuffle=True):
    # Create the data and label variables so we can use them in the computation
    img_var = Variable(torch.FloatTensor(img),requires_grad=False)
    #lbl = Variable(torch.LongTensor(lbl))

    # Call a forward pass on the data
    output = model(img_var)
    
    # Run quick accuracy check
    #print("GT Label: ",lbl.data.numpy())
    #print("Output: ",output.data.numpy())
    
    # Get the max index from the ith softmax array
    guess = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    cmat[int(lbl[0]),int(guess[0])] += 1    

    # if index == gt label call it correct
    if int(lbl[0]) == int((output.data.max(1, keepdim=True)[1][0])):
        correct_cnt += 1
    total_cnt += 1
    if total_cnt%100 == 0:
        print("Accuracy@{}: {}".format(total_cnt,correct_cnt/float(total_cnt)))
        print("Confusion Matrix: \n",cmat)
print("\nFINAL Accuracy: {}".format(correct_cnt/float(total_cnt)))


