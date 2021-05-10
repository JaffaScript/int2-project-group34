import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import time
from net import Net

# DEFINE HYPERPAREMETERS AND ASSIGN DEFAULTS #
no_of_epochs = 20
batch_size = 4
learning_rate = 0.001
momentum = 0.90
#

print("\nPlease enter hyper parameters:\n-------------------")
no_of_epochs = int(input("How many epochs: "))
learning_rate = float(input("Learning rate: "))
momentum = float(input("Momentum: "))
print()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: %s" % device)
print()

print("** Creating transforms.. **")
# CREATE TRANSFORMS #   
transforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

transforms_testing = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
#

print("** Defining categories.. **")
# DEFINE CLASSES #
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#

print("** Importing and preparing data.. **")
# IMPORT DATASETS #

dataset_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
loader_training = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=True, num_workers=0)

dataset_testing = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_testing)
loader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=batch_size, shuffle=False, num_workers=0)
#

print("** LOADING COMPLETE! **\n")
print("** Building net model.. **")
# DESIGN NET MODEL #


net = Net()
net.to(device)

#

# DEFINE LOSS FUNCTION & OPTIMIZER #
criterion = nn.CrossEntropyLoss()   # because it is a multi-class problem
optimiser = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#

# TRAIN NN #

t0 = time.perf_counter()   # DEBUG ONLY.
for epoch in range(no_of_epochs):
    running_loss = 0.0
    for i, data in enumerate(loader_training, 0):
        inputs, labels = data[0].to(device), data[1].to(device) # get the inputs; data is a list of [inputs, labels]
        optimiser.zero_grad() # zero the parameter gradients

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        # print stats
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

t1 = time.perf_counter() - t0
print("\nFinished training in %f seconds." % t1)
#

# SAVE TRAINED MODEL #
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(loader_testing)
images, labels = dataiter.next()

## can save and re load net
## loading code:
##  net = Net()
##  net.load_state_dict(torch.load(PATH))

# TEST ON DATASET #
correct = 0
total = 0
with torch.no_grad():
    for data in loader_testing:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))