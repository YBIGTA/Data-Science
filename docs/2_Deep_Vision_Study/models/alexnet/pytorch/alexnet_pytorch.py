import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class VGG16(nn.Module):
    def __init__(self,n_out=10):
        super(VGG16,self).__init__()
        layers = []
        before_channels = 3
        current_channels = 64
        for i in range(5):
            num_repeat = 2 if i<2 else 3
            for j in range(num_repeat):
                in_channels = before_channels if j==0 else current_channels
                out_channels = int(current_channels)
                layers.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = True))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            before_channels = current_channels
            current_channels *= 2
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
                            nn.Linear(7*7*512,4096),
                            nn.ReLU(),
                            nn.Linear(4096,4096),
                            nn.ReLU(),
                            nn.Linear(4096, n_out)
                        )
    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,512*7*7)
        out = self.classifier(x)
        return out

class VGG16bn(nn.Module):
    def __init__(self):
        super(VGG16bn,self).__init__()
        layers = []
        before_channels = 3
        current_channels = 64
        for i in range(3):
            num_repeat = 2 if i<2 else 3
            for j in range(num_repeat):
                in_channels= before_channels if j==0 else current_channels
                out_channels = int(current_channels)
                layers.append(nn.Conv2d(in_channels = before_channesl, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            before_channels = current_channels
            current_channels /= 2
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
                            nn.Linear(7*7*512,4096),
                            nn.Batchnorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096,4096),
                            nn.batchnorm1d(4096),
                            nn.ReLU(),
                            nn.Linear(4096,n_out)
                        )

    def forward(self,x):
        x = self.encoder(x)
        x = x.view(-1,7*7*512)
        out = self.classifier(x)
        return out


class AlexNetOriginal(nn.Module):
    def __init__(self, n_out=10):
        super(AlexNetOriginal,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride =4, bias = True),
                    nn.ReLU(),
                    nn.LocalResponseNorm(size=5, alpha=1e-5, beta=0.75, k=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=96, out_channels = 256, kernel_size = 5, stride = 1, padding=2, bias = True),
                    nn.ReLU(),
                    nn.LocalResponseNorm(size=5, alpha=1e-5, beta=0.75, k=2)
                    )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias = True),
                    nn.ReLU()
                    )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias = True),
                    nn.ReLU()
                    )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias = True),
                    nn.ReLU(),
                    nn.LocalResponseNorm(size=5, alpha=1e-5, beta=0.75, k=2)
                    )
        self.fc1 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features = 256*6*6, out_features = 4096, bias = True),
                    nn.ReLU(),
                    )
        self.fc2 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features = 4096, out_features = 4096, bias = True),
                    nn.ReLU()
                    )
        self.fc3 = nn.Linear(in_features = 4096, out_features = n_out)

    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1,256*6*6)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


class AlexNetBn(nn.Module):
    def __init__(self, n_out=10):
        super(AlexNetBn,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 11, stride =4, bias = False),
                    nn.BatchNorm2d(96),
                    nn.ReLU(),
                    )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(in_channels=96, out_channels = 256, kernel_size = 5, stride = 1, padding=2, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU()
                    )
        self.conv4 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU()
                    )
        self.conv5 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    )
        self.fc1 = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(in_features = 256*6*6, out_features = 4096, bias = True),
                    nn.ReLU(),
                    )
        self.fc2 = nn.Sequential(
                   nn.Dropout(),
                   nn.Linear(in_features = 4096, out_features = 4096, bias = True),
                   nn.ReLU()
                    )
        self.fc3 = nn.Linear(in_features = 4096, out_features = 10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = x.view(-1,256*6*6)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out
