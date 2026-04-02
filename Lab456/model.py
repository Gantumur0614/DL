import torch.nn.functional as F
import torch.nn as nn
import torch
import torch 
import torch.nn as nn 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(self,  in_channels,  out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):

        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1))

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        #self.avgpool = nn.AdaptiveAvgPool2d(kernel_size=5, stride=3)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x) # updated added sigmoid 


class InceptionModel(nn.Module):
    def __init__(self, aux=False, residual=True, num_classes=1000):
        super(InceptionModel, self).__init__()
        self.aux = aux
        self.residual = residual

        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)

        self.incept3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.incept3b = InceptionBlock(256, 128, 128, 192, 32, 112, 80)

        self.incept4a = InceptionBlock(512, 192, 96, 208, 16, 48, 64)
        self.incept4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)

        if self.aux:
            self.aux_classifier = InceptionAux(512, num_classes)

        self.incept5a = InceptionBlock(512, 256, 160, 320, 32, 128, 128)
        self.incept5b = InceptionBlock(832, 128, 112, 256, 32, 64, 64)

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * 4 * 4, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.incept3a(x)
        x = self.incept3b(x)

        residual = self.maxpool(x)

        x = self.incept4a(residual)

        x = self.incept4b(x)

        if self.residual:
            x += residual

        if self.aux and self.training:
            aux_out = self.aux_classifier(x)

        residual = self.maxpool(x)

        x = self.incept5a(residual)
        x = self.incept5b(x)

        if self.residual:
            x += residual

        x = F.adaptive_avg_pool2d(x, (4, 4))

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)    # added sigmoid 

        if self.aux and self.training:
            return x, aux_out
        return x
