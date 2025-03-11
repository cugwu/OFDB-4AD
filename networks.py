from typing import Any
from functools import partial
import torch.nn as nn

from models.resnet import _resnet, Bottleneck, BasicBlock, ResNet
from models.ffc_resnet import FFCBottleneck, FFCBasicBlock, FFCResNet
from models.gfnet import _gfnet_h, _gfnet

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           # FFC
           'ffc_resnet18', 'ffc_resnet34', 'ffc_resnet26', 'ffc_resnet50',
           'ffc_resnet101', 'ffc_resnet152', 'ffc_resnet200', 'ffc_resnext50_32x4d', 'ffc_resnext101_32x8d',
           'ffc_wide_resnet50_2', 'ffc_wide_resnet101_2',
           # Global Filter
           'gfnet_xs', 'gfnet_h_ti', 'gfnet_h_s', 'gfnet_h_b', 'gfnet_ti', 'gfnet_s', 'gfnet_b']

# Resnet
def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


# Fast-Fourier-Convolution

def ffc_resnet18(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def ffc_resnet34(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def ffc_resnet26(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-26 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBottleneck, [2, 2, 2, 2], **kwargs)
    return model


def ffc_resnet50(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def ffc_resnet101(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def ffc_resnet152(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def ffc_resnet200(pretrained=False, **kwargs):
    """Constructs a FFT ResNet-200 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FFCResNet(FFCBottleneck, [3, 24, 36, 3], **kwargs)
    return model


def ffc_resnext50_32x4d(pretrained=False, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = FFCResNet(FFCBottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ffc_resnext101_32x8d(pretrained=False, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = FFCResNet(FFCBottleneck, [3, 4, 32, 3], **kwargs)

    return model


def ffc_wide_resnet50_2(pretrained=False, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model = FFCResNet(FFCBottleneck, [3, 4, 6, 3], **kwargs)

    return model


def ffc_wide_resnet101_2(pretrained=False, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    model = FFCResNet(FFCBottleneck, [3, 4, 23, 3], **kwargs)

    return model

# Global Filter - Transformer

def gfnet_xs(pretrained: bool = False, **kwargs):
    return _gfnet(img_size=kwargs['img_size'], patch_size=16, embed_dim=384, depth=12, mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  pretrained=pretrained, **kwargs)


def gfnet_ti(pretrained: bool = False, **kwargs):
    return _gfnet(img_size=kwargs['img_size'], patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  pretrained=pretrained, **kwargs)


def gfnet_s(pretrained: bool = False, **kwargs):
    return _gfnet(img_size=kwargs['img_size'], patch_size=16, embed_dim=384, depth=19, mlp_ratio=4, drop_path_rate=0.15,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  pretrained=pretrained, **kwargs)


def gfnet_b(pretrained: bool = False, **kwargs):
    return _gfnet(img_size=kwargs['img_size'], patch_size=16, embed_dim=512, depth=19, mlp_ratio=4, drop_path_rate=0.25,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                  pretrained=pretrained, **kwargs)


def gfnet_h_ti(pretrained: bool = False, **kwargs):
    return _gfnet_h('gfnet_h_ti', patch_size=4, embed_dim=[64, 128, 256, 512], depth=[3, 3, 10, 3],
                    mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    drop_path_rate=0.1, pretrained=pretrained, **kwargs)


def gfnet_h_s(pretrained: bool = False, **kwargs):
    return _gfnet_h('gfnet_h_s', patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 10, 3],
                    mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    drop_path_rate=0.2, init_values=1e-5, pretrained=pretrained, **kwargs)


def gfnet_h_b(pretrained: bool = False, **kwargs):
    return _gfnet_h('gfnet_h_b', patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 27, 3],
                    mlp_ratio=[4, 4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    drop_path_rate=0.4, init_values=1e-6, pretrained=pretrained, **kwargs)
