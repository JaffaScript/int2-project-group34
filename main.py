import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = True
        # Constraints for layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Constraints for layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Constraints for layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.batch3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Constraints for layer 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.batch4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Defining the Linear layer
        self.fc = nn.Linear(128 * 2 * 2, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        if self.shape:
            print("ORIG " + str(x.shape))

        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        if self.shape:
            print("1 " + str(out.shape))

        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        if self.shape:
            print("2 " + str(out.shape))

        # Conv 3 
        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        if self.shape:
            print("3 " + str(out.shape))

        # Conv 4 
        out = self.conv4(out)
        out = self.batch4(out)
        out = self.relu4(out)
        out = self.pool4(out)

        if self.shape:
            print("4 " + str(out.shape))

        # Fitting
        out = out.view(out.size(0), -1)

        if self.shape:
            print("6 " + str(out.shape))

        # Linear Layer
        out = self.fc(out)
        out = self.fc2(out)
        out = self.fc3(out)

        if self.shape:
            print("7 " + str(out.shape))
            self.shape = False

        return out

def main():
    # Defining hyperparameters
    num_epochs = int(input("Enter number of cyles through the dataset: "))
    batch_size = 128
    learning_rate = 0.005

    # Defining device (GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # BEGIN TRAINING
    print('Training network...')

    # Transforming Data
    # Dataset has PILImage images with range 0, 1
    # Transformed to Tensors of normalised range -1, 1
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Downloading datasets
    training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
    testing_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    training_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Assigning classes
    classes = training_data.classes

    # Defining model
    model = Net().to(device)

    # Defining criteria and optimiser
    # Since this is a multiclass classification problem, we use CrossEntropyLoss function
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    ###### BEGIN TRAINING ######
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(training_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device)

            # Forward pass and calculate the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass, and empty the gradient + optimise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 150 == 149:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished training.')

    # Save trained network
    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)
    ###### FINISH TRAINING ######

    ###### BEGIN TESTING ######
    print('Testing...')

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        num_correct_classes = [0 for i in range(10)]
        num_samples_classes = [0 for i in range(10)]

        for data in testing_dataloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()

            for i in range(4):
                label = labels[i]
                prediction = predicted[i]
                if (label == prediction):
                    num_correct_classes[label] += 1
                num_samples_classes[label] += 1

        accuracy = 100.0 * num_correct / num_samples
        print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

        for i in range(10):
            accuracy = 100.0 * num_correct_classes[i] / num_samples_classes[i]
            print(f'Accuracy of class {classes[i]}: {accuracy} %')

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        num_correct_classes = [0 for i in range(10)]
        num_samples_classes = [0 for i in range(10)]

        for data in training_data:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            num_samples += labels.size(0)
            num_correct += (predicted == labels).sum().item()

            for i in range(4):
                label = labels[i]
                prediction = predicted[i]
                if (label == prediction):
                    num_correct_classes[label] += 1
                num_samples_classes[label] += 1

        accuracy = 100.0 * num_correct / num_samples
        print(f'Accuracy of the network on the 40000 training images: {accuracy} %')

        for i in range(10):
            accuracy = 100.0 * num_correct_classes[i] / num_samples_classes[i]
            print(f'Accuracy of class {classes[i]}: {accuracy} %')

    print('Finished testing.')
    ###### FINISH TESTING ######

if __name__ == '__main__':
    main()