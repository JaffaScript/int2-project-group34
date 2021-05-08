import numpy as np

import pandas as pandas
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: %s" % device)

print("** Creating transforms.. **")
# CREATE TRANSFORMS #
t1 = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#

print("** Defining categories.. **")
# DEFINE CLASSES #
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#

print("** Importing and preparing data.. **")
# IMPORT DATASETS #

bsize = 4  # specifies number of images to load per epoch

dataset_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=t1)
loader_training = torch.utils.data.DataLoader(dataset_training, batch_size=bsize, shuffle=True, num_workers=0)

dataset_testing = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=t1)
loader_testing = torch.utils.data.DataLoader(dataset_testing, batch_size=bsize, shuffle=False, num_workers=0)
#

print("** LOADING COMPLETE! **\n")
print("** Building net model.. **")
# DESIGN NET MODEL #
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.to(device)
#

# DEFINE LOSS FUNCTION & OPTIMIZER #
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#

# TRAIN NN #
for epoch in range(2):
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

print("Finished training.")
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