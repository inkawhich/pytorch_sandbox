# NAI

# Here we will train a model and save it with: 'torch.save(model, './mnist_saved_model.pth')'. In another
#   file we will show how to load for testing and load for retraining

from __future__ import print_function,division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import struct
import random
import matplotlib.pyplot as plt


mnist_train_data = []
mnist_train_labels = []

# Inspired by https://gist.github.com/akesling/5358964
# Load everything in some numpy arrays
with open("./data/train-labels-idx1-ubyte", 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
with open("./data/train-images-idx3-ubyte", 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

# Flatten the 28x28 images into 1x784
# Read in training data and labels
for i in img:
    mnist_train_data.append(i.flatten())
for int_label in lbl:
    mnist_train_labels.append(int_label)

print("train data array: ",np.array(mnist_train_data).shape)
print("train labels array: ",np.array(mnist_train_labels).shape)

class MNIST_Dataset(object):
    def __init__(self, train_data, train_labels):
        self.images = train_data
        self.labels = train_labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)

    def make_batch(self, iterable, batch_size=1):
        length = len(iterable)
        for index in range(0, length, batch_size):
            yield iterable[index:min(index + batch_size, length)]

    def read(self, batch_size, shuffle=True):
        order = list(range(len(self)))
        if shuffle:
            random.shuffle(order)
        for batch in self.make_batch(order, batch_size):
            images, labels = [], []
            for index in batch:
                image, label = self[index]
                images.append(np.expand_dims(image.reshape((28,28)),axis=0))
                labels.append(label)
            yield np.stack(images).astype(np.float32), np.stack(labels).astype(np.int32).reshape((batch_size,))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320) # Reshape tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


################################
# Run Training
################################
b_size = 50
num_epochs = 2
train_dataset = MNIST_Dataset(mnist_train_data, mnist_train_labels)

model = Net()
optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)


print("\nTraining...")
# Training Loop
model.train() # I think this sets the dropouts in train mode

iteration = 0
for e in range(num_epochs):
    for img,lbl in train_dataset.read(batch_size=b_size, shuffle=True):
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
        if iteration%100 == 0:
            correct_cnt = 0
            for i in range(b_size):
                # Get the max index from the ith softmax array
                guess = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                # if index == gt label call it correct
                if int(lbl[i]) == int((output.data.max(1, keepdim=True)[1][i])):
                    correct_cnt += 1
            print("Iter: {}, Accuracy: {}".format(iteration,correct_cnt/float(b_size)))

        # Compute the loss
        loss = F.nll_loss(output,lbl)
        # Compute the gradients in the backward step
        loss.backward()
        # Run the optimizer and update the weights
        optimizer.step()
        iteration += 1

# Saving Model
print("Saving Model...")
torch.save(model, './mnist_saved_model.pth')


