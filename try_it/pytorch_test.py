import torch
import torchvision
from torchvision import transforms, datasets

train = datasets.MNIST("", train=True, download=True,
                       transform=transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))]))
test = datasets.MNIST("", train=False, download=True,
                      transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5,), (0.5,))]))

train_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_set = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)