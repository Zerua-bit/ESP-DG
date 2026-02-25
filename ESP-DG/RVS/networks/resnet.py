import math
import os
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from .LayerDiscriminator import LayerDiscriminator
hub_dir = torch.hub.get_dir()
model_dir = os.path.join('/workspace/data', 'checkpoints')


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

in_affine = False


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, ini=3,
                 domains=3, domain_discriminator_flag=0, grl=0, lambd=0., drop_percent=0.33, dropout_mode=0, wrs_flag=0, recover_flag=0, layer_wise_flag=0, V=40):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(ini, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, name='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, name='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, name='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, name='layer4')

        self.V = V
        self.domain_discriminator_flag = domain_discriminator_flag
        self.drop_percent = drop_percent
        self.dropout_mode = dropout_mode

        self.recover_flag = recover_flag
        self.layer_wise_flag = layer_wise_flag
        self.layer_channels = [64, 64, 128, 256, 512]
        self.domain_discriminators = nn.ModuleList([
            LayerDiscriminator(
                num_channels=self.layer_channels[layer],
                num_classes=domains,
                grl=grl,
                reverse=True,
                lambd=lambd,
                wrs_flag=wrs_flag,
            )
            for i, layer in enumerate([0, 1, 2, 3, 4])])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, name='layern'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def perform_dropout(self, feature, domain_labels, layer_index, layer_dropout_flag, step):
        domain_output = None
        if self.domain_discriminator_flag and self.training:
            index = layer_index
            G_t = (2 / math.pi) * math.atan(step / self.V)
            percent = self.drop_percent * G_t
            domain_output, domain_mask = self.domain_discriminators[index](
                feature.clone(),
                domain_labels,
                percent=percent,
            )
            if self.recover_flag:
                domain_mask = domain_mask * domain_mask.numel() / domain_mask.sum()
            if layer_dropout_flag:
                feature = feature * domain_mask
        return feature, domain_output

    def forward(self, x, domain_labels=None, layer_drop_flag=None, step=0):
        sfs = []
        domain_outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.training:
            x, domain_output = self.perform_dropout(x, domain_labels, layer_index=0,
                                                    layer_dropout_flag=layer_drop_flag[0], step=step)
            if domain_output is not None:
                domain_outputs.append(domain_output)

        sfs.append(x)
        x = self.maxpool(x)

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if self.training:
                x, domain_output = self.perform_dropout(x, domain_labels, layer_index=i + 1,
                                                        layer_dropout_flag=layer_drop_flag[i + 1], step=step)
                if domain_output is not None:
                    domain_outputs.append(domain_output)
            sfs.append(x)

        return sfs, domain_outputs


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock,[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet34-333f7ec4.pth')), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'resnet50-19c8e357.pth')), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
