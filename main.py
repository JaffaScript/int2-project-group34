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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

    def forward(self, x):
        #x = self.layer1(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():

    # Defining hyperparameters
    num_epochs = int(input("Enter number of cyles through the dataset: "))
    batch_size = 4
    learning_rate = 0.001

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
            if i % 2000 == 1999:    # print every 2000 mini-batches
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

            for i in range(batch_size):
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

    print('Finished testing.')
    ###### FINISH TESTING ######

if __name__ == '__main__':
    main()