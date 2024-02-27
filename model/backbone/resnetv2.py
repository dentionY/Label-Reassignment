### 在分析得到若干条经验后自行设计tinynet

from __future__ import absolute_import, division, print_function
import torch.nn as nn
from ..module.activation import act_layers

backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

## FX GRAPH
from torch.quantization import quantize_fx

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, activation="ReLU"):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = act_layers(activation)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        #if self.downsample is not None:
        #    residual = self.downsample(x)

        #out += residual
        out = self.act(out)

        return out

class ResNet(nn.Module):
    resnet_spec = {
        18: (BasicBlock, [1, 1, 1, 1]),
        34: (BasicBlock, [3, 4, 6, 3]),
    }

    def __init__(
        self, depth, out_stages=(1, 2, 3, 4), activation="ReLU", pretrain=False
    ):
        super(ResNet, self).__init__()
        if depth not in self.resnet_spec:
            raise KeyError("invalid resnet depth {}".format(depth))
        assert set(out_stages).issubset((1, 2, 3, 4))
        self.activation = activation
        block, layers = self.resnet_spec[depth]
        self.depth = depth
        self.inplanes = 64
        self.out_stages = out_stages

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = act_layers(self.activation)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #self.layer1 = self._make_layer(block, 64, layers[0])
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 160, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.init_weights(pretrain=pretrain)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if planes != 32 or stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, activation=self.activation)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=self.activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        #x = self.maxpool(x)
        x = self.conv2(x)
        x = self.bn2(x)

        output = []
        for i in range(1, 5):
            if i <= self.out_stages[-1]:  # 不再往后构造
                res_layer = getattr(self, "layer{}".format(i))
                x = res_layer(x)
                if i in self.out_stages:
                    output.append(x)
        return tuple(output)

    def init_weights(self, pretrain=True):
        for m in self.modules():
            if self.activation == "LeakyReLU":
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
