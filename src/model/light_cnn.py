from torch import nn 
import torch

class LightCNN(nn.Module):
    def __init__(self, conv_blocks, max_pooling_conf, mlp1_conf, mlp2_conf):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            Convblock(in_channels, out_channels, kernel_size, stride, padding, use_batchnorm)
            for (in_channels, out_channels, kernel_size, stride, padding, use_batchnorm) in conv_blocks
        ])

        self.max_pool = nn.MaxPool2d(**max_pooling_conf)
        self.mlp1 = MLPblock(**mlp1_conf)
        self.mlp2 = nn.Linear(**mlp2_conf)

    def forward(self, data_object):
        data_object = self.conv_blocks[0](data_object)
        data_object = self.max_pool(data_object)
        data_object = self.conv_blocks[1](data_object)
        data_object = self.conv_blocks[2](data_object)
        data_object = self.max_pool(data_object)
        data_object = self.conv_blocks[3](data_object)
        data_object = self.conv_blocks[4](data_object)
        data_object = self.max_pool(data_object)
        data_object = self.conv_blocks[5](data_object)
        data_object = self.conv_blocks[6](data_object)
        data_object = self.conv_blocks[7](data_object)
        data_object = self.conv_blocks[8](data_object)
        data_object = self.max_pool(data_object)
        data_object = data_object.view(data_object.size(0), -1)
        data_object = self.mlp1(data_object)
        data_object = self.mlp2(data_object)
        return data_object



class MFM(nn.Module): 
    def __init__(self):
        super().__init__()

    def forward(self, data_object):
        assert data_object.dim() in (2, 4), "Input must be 4D or 2D tensor [(batch_size, channels, height, width) or (batch_size, features)]"
        if data_object.dim() == 2:
            data_object = data_object.view(data_object.size(0), 2, -1)
            return data_object.max(dim=1).values
        elif data_object.dim() == 4:
            return torch.maximum(data_object[:, ::2, :, :], data_object[:, 1::2, :, :])
    
class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_batchnorm):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mfm = MFM()
        self.bn = None
        if use_batchnorm:
            self.bn = nn.BatchNorm2d(out_channels // 2)

    def forward(self, data_object):
        data_object = self.conv(data_object)
        data_object = self.mfm(data_object)
        if self.bn is not None:
            data_object = self.bn(data_object)
        return data_object
    
class MLPblock(nn.Module):
    def __init__(self, in_features, out_features, use_batchnorm=True):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.mfm = MFM()
        self.bn = None
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(out_features // 2)

    def forward(self, data_object):
        data_object = self.fc(data_object)
        data_object = self.mfm(data_object)
        if self.bn is not None:
            data_object = self.bn(data_object)
        return data_object
    
