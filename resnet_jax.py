from typing import Any, Callable, List
from flax import linen as nn
import einops


def conv3x3(out_planes: int, stride: int = 1, kernel_init=nn.initializers.kaiming_normal):
    return nn.Conv(out_planes, kernel_size=(3, 3), strides=(stride, stride), use_bias=False, kernel_init=kernel_init)


def conv1x1(out_planes: int, stride: int = 1, kernel_init=nn.initializers.kaiming_normal):
    return nn.Conv(out_planes, kernel_size=(1, 1), strides=(stride, stride), use_bias=False, kernel_init=kernel_init)


class BasicBlock(nn.Module):
    planes: int
    stride: int = 1
    norm_layer: Callable = None

    @nn.compact
    def __call__(self, x):
        in_planes = x.shape[-1]
        identity = x

        out = conv3x3(self.planes, self.stride)(x)
        out = self.norm_layer(self.planes)(out)
        out = nn.relu(out)

        out = conv3x3(self.planes)(out)
        out = self.norm_layer(self.planes)(out)

        if self.stride != 1 or self.planes != in_planes:
            identity = conv1x1(self.planes, self.stride)
            identity = self.norm_layer(self.planes)(identity)

        out += identity
        out = nn.relu(out)

        return out


class BasicLayer(nn.Module):
    planes: int
    blocks: int
    stride: int = 1
    norm_layer: Callable = None

    @nn.compact
    def __call__(self, x):
        for i in range(self.blocks):
            x = BasicBlock(self.planes, self.stride if i == 0 else 1, self.norm_layer)(x)
        return x


class ResNet(nn.Module):
    layers: List[int]
    num_classes: int = 10
    zero_init_residual: bool = False
    norm_layer: Callable = nn.BatchNorm

    @nn.compact
    def __call__(self, x):
        inplanes = 64
        x = conv3x3(inplanes)(x)
        x = self.norm_layer()(x)
        x = nn.relu(x)
        x = BasicLayer(64, self.layers[0])(x)
        x = BasicLayer(128, self.layers[1], 2, self.norm_layer)(x)
        x = BasicLayer(256, self.layers[2], 2, self.norm_layer)(x)
        x = BasicLayer(512, self.layers[3], 2, self.norm_layer)(x)
        x = nn.avg_pool(x, x.shape[1:3])
        x = einops.rearrange(x, 'B () () C -> B C')
        x = nn.Dense(self.num_classes)(x)
        return x


def resnet18(**kwargs: Any) -> ResNet:
    return ResNet([2, 2, 2, 2], **kwargs)
