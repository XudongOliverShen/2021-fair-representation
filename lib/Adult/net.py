import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dim_out=32):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(64*4*4, dim_out)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x): # [N,3,64,64]
        out = F.leaky_relu(self.bn1(self.conv1(x))) # [N,64,64,64]
        out = self.layer1(out) # [N,64,32,32]
        out = self.layer2(out) # [N,64,16,16]
        out = self.layer3(out) # [N,64,8,8]
        out = self.layer4(out) # [N,64,4,4]
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(dim_out=32):
    return ResNet(BasicBlock, [2,2,2,2], dim_out)

class BasicBlock_transposed(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_transposed, self).__init__()
        if stride==1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_planes, self.expansion*planes, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

class ResNet_transposed(nn.Module):
    def __init__(self, block, num_blocks, dim_in=32+1):
        super(ResNet_transposed, self).__init__()
        self.in_planes = 64

        self.linear = nn.Linear(dim_in, 64*4*4)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 8, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 4, num_blocks[3], stride=2)
        self.conv5 = nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, h): # [N,33]
        out = self.linear(h) # [N, 1024]
        out = out.view(out.shape[0],64,4,4) # [N,64,4,4]
        out = self.layer1(out) # [N,32,8,8]
        out = self.layer2(out) # [N,16,16,16]
        out = self.layer3(out) # [N,8,32,32]
        out = self.layer4(out) # [N,4,64,64]
        out = self.conv5(out) # [N,3,64,64]
        out = nn.Sigmoid()(out)
        return out

def ResNet18_transposed(dim_in=32+1):
    return ResNet_transposed(BasicBlock_transposed, [2,2,2,2], dim_in)



class DenseBlock(nn.Module):
    def __init__(self, dim_in=166, dim_hidden=256, dim_out=4, sigmoid=True):
        super(DenseBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.fc1 = nn.Linear(in_features=dim_in, out_features=dim_hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(in_features=dim_hidden, out_features=dim_hidden, bias=False)
        self.bn2 = nn.BatchNorm1d(1)
        self.fc3 = nn.Linear(in_features=dim_hidden, out_features=dim_hidden, bias=False)
        self.bn3 = nn.BatchNorm1d(1)
        self.fc4 = nn.Linear(in_features=dim_hidden, out_features=dim_hidden, bias=False)
        self.bn4 = nn.BatchNorm1d(1)
        self.fc5 = nn.Linear(in_features=dim_hidden, out_features=dim_hidden, bias=False)
        self.bn5 = nn.BatchNorm1d(1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.head = nn.Linear(dim_hidden, dim_out)
        self.sigmoid = sigmoid

        self.h_o2 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False),
            nn.BatchNorm1d(1)
        )
        self.h_o3 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False),
            nn.BatchNorm1d(1)
        )
        self.h_o4 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False),
            nn.BatchNorm1d(1)
        )
        self.h_o5 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False),
            nn.BatchNorm1d(1)
        )

    def forward(self, h):
        # input embedding: [256,32]
        N = h.shape[0]
        h = h.reshape([N, 1, -1])

        o1 = self.relu( self.bn1(self.fc1(h))  )
        o2 = self.relu( self.bn2(self.fc2(o1)) ) + self.h_o2(h)
        o3 = self.relu( self.bn3(self.fc3(o2)) ) + self.h_o3(h)
        o4 = self.relu( self.bn4(self.fc4(o3)) ) + self.h_o4(h)
        o5 = self.relu( self.bn5(self.fc5(o4)) ) + self.h_o5(h)
        o6 = self.head( o5 )

        if self.sigmoid:
            return torch.sigmoid(o6.reshape([N, -1]))
        else:
            return o6.reshape([N,-1])