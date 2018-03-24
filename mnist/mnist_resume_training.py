# NAI

# This is meant to be a BARE-BONES mnist tutorial. Just train and test, nothing else.

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
mnist_test_data = []
mnist_test_labels = []

# Inspired by https://gist.github.com/akesling/5358964
# Load everything in some numpy arrays
with open(os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/train-labels-idx1-ubyte"), 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
with open(os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/train-images-idx3-ubyte"), 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
with open(os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/t10k-labels-idx1-ubyte"), 'rb') as ftstlbl:
    magic, num = struct.unpack(">II", ftstlbl.read(8))
    tst_lbl = np.fromfile(ftstlbl, dtype=np.int8)
with open(os.path.join(os.path.expanduser('~'),"DukeML/datasets/mnist/t10k-images-idx3-ubyte"), 'rb') as ftstimg:
    magic, num, rows, cols = struct.unpack(">IIII", ftstimg.read(16))
    tst_img = np.fromfile(ftstimg, dtype=np.uint8).reshape(len(tst_lbl), rows, cols)

# Flatten the 28x28 images into 1x784
# Read in training data and labels
for i in img:
    mnist_train_data.append(i.flatten())
for int_label in lbl:
    mnist_train_labels.append(int_label)
# Read in test data and labels
for i in tst_img:
    mnist_test_data.append(i.flatten())
for int_label in tst_lbl:
    mnist_test_labels.append(int_label)

print("train data array: ",np.array(mnist_train_data).shape)
print("train labels array: ",np.array(mnist_train_labels).shape)
print("test data array: ",np.array(mnist_test_data).shape)
print("test labels array: ",np.array(mnist_test_labels).shape)

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
print("\nTraining...")

b_size = 50
num_epochs = 1
train_dataset = MNIST_Dataset(mnist_train_data, mnist_train_labels)

# Load trained model for finetuning
model = Net()
model = torch.load('./mnist_saved_model.pth')
model.train() 
optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.5)

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


################################
# Run Testing
################################
print("\nTesting...")
correct_cnt = 0
total_cnt = 0
# Construct the testing dataset
test_dataset = MNIST_Dataset(mnist_test_data, mnist_test_labels)
# Set the dropout layers to test mode
model.eval() 
for img,lbl in test_dataset.read(batch_size=1, shuffle=True):
    # Create the data and label variables so we can use them in the computation
    img = Variable(torch.FloatTensor(img),requires_grad=False)
    lbl = Variable(torch.LongTensor(lbl))
    # Call a forward pass on the data
    output = model(img)
    # if index == gt label call it correct
    if int(lbl[0]) == int((output.data.max(1, keepdim=True)[1][0])):
        correct_cnt += 1
    total_cnt += 1
    if total_cnt%1000 == 0:
        print("Accuracy: {}".format(correct_cnt/float(total_cnt)))

print("\nFINAL Accuracy: {}".format(correct_cnt/float(total_cnt)))
