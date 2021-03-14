import torch
import torchvision
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import helper
import datetime


def get_train_test_set():
    train = datasets.MNIST("", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.5,), (0.5,))]))
    test = datasets.MNIST("", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5,), (0.5,))]))
    train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    test_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    return train_set, test_set


def train_data(model, epochs, trainloader, criterion, optimizer):
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")


class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
    train_set, test_set = get_train_test_set()
    model = NeuralNet().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_data(model, 5, train_set, criterion, optimizer)
    print(datetime.datetime.now() - begin_time)
