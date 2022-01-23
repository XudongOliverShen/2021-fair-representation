import torch
import torch.nn as nn

# -------------------------------------- sphere network Begin --------------------------------------
class Block(nn.Module):
    def __init__(self, planes):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu2 = nn.PReLU(planes)

    def forward(self, x):
        return x + self.prelu2(self.conv2(self.prelu1(self.conv1(x))))

class sphere(nn.Module):
    def __init__(self, type=20, is_gray=False, emd_dim=512):
        super(sphere, self).__init__()
        block = Block

        if type is 20:
            layers = [1, 2, 4, 1] # [1, 2, 4, 1]
        elif type is 36:
            layers = [2, 4, 8, 2]
        elif type is 64:
            layers = [3, 7, 16, 3]
        elif type is 2:
            layers = [2, 2, 2, 2]
        elif type is 3:
            layers = [3, 3, 3, 3]
        elif type == 3443:
            layers = [3, 4, 4, 3]
        elif type == 3463:
            layers = [3, 4, 6, 3]
        else:
            raise ValueError('sphere' + str(type) + " IS NOT SUPPORTED! (sphere20 or sphere64)")
        filter_list = [3, 64, 128, 256, 512]
        if is_gray:
            filter_list[0] = 1

        self.layer1 = self._make_layer(block, filter_list[0], filter_list[1], layers[0], stride=2)
        self.layer2 = self._make_layer(block, filter_list[1], filter_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(block, filter_list[2], filter_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(block, filter_list[3], filter_list[4], layers[3], stride=2)

        self.fc = nn.Linear(512 * 7 * 7, emd_dim)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)


    def _make_layer(self, block, inplanes, planes, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(inplanes, planes, 3, stride, 1))
        layers.append(nn.PReLU(planes))
        for i in range(blocks):
            layers.append(block(planes))

        return nn.Sequential(*layers)

    def forward(self, x): # [512, 3, 112, 112]
        x = self.layer1(x) # [512, 64, 56, 56]
        x = self.layer2(x) # [512, 128, 28, 28]
        x = self.layer3(x) # [512, 256, 14, 14]
        x = self.layer4(x) # [512, 512, 7, 7]

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

# -------------------------------------- sphere network END --------------------------------------
