# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmcv.ops.fused_bias_leakyrelu import (FusedBiasLeakyReLU,
                                           fused_bias_leakyrelu)
# from basicsr.ops.fused_act import FusedLeakyReLU, fused_leaky_relu
from basicsr.ops.upfirdn2d import upfirdn2d
import torch.nn.functional as F

@BACKBONES.register_module()
class GLEANStyleGANv2FuC(nn.Module):
    r"""GLEAN (using StyleGANv2) architecture for super-resolution.

    Paper:
        GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution,
        CVPR, 2021

    This method makes use of StyleGAN2 and hence the arguments mostly follow
    that in 'StyleGAN2v2Generator'.

    In StyleGAN2, we use a static architecture composing of a style mapping
    module and number of covolutional style blocks. More details can be found
    in: Analyzing and Improving the Image Quality of StyleGAN CVPR2020.

    You can load pretrained model through passing information into
    ``pretrained`` argument. We have already offered official weights as
    follows:

    - styelgan2-ffhq-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth  # noqa
    - stylegan2-horse-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-horse-config-f-official_20210327_173203-ef3e69ca.pth  # noqa
    - stylegan2-car-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-car-config-f-official_20210327_172340-8cfe053c.pth  # noqa
    - styelgan2-cat-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-cat-config-f-official_20210327_172444-15bc485b.pth  # noqa
    - stylegan2-church-config-f: http://download.openmmlab.com/mmgen/stylegan2/official_weights/stylegan2-church-config-f-official_20210327_172657-1d42b7d1.pth  # noqa

    If you want to load the ema model, you can just use following codes:

    .. code-block:: python

        # ckpt_http is one of the valid path from http source
        generator = StyleGANv2Generator(1024, 512,
                                        pretrained=dict(
                                            ckpt_path=ckpt_http,
                                            prefix='generator_ema'))

    Of course, you can also download the checkpoint in advance and set
    ``ckpt_path`` with local path. If you just want to load the original
    generator (not the ema model), please set the prefix with 'generator'.

    Note that our implementation allows to generate BGR image, while the
    original StyleGAN2 outputs RGB images by default. Thus, we provide
    ``bgr2rgb`` argument to convert the image space.

    Args:
        in_size (int): The size of the input image.
        out_size (int): The output size of the StyleGAN2 generator.
        img_channels (int): Number of channels of the input images. 3 for RGB
            image and 1 for grayscale image. Default: 3.
        rrdb_channels (int): Number of channels of the RRDB features.
            Default: 64.
        num_rrdbs (int): Number of RRDB blocks in the encoder. Default: 23.
        style_channels (int): The number of channels for style code.
            Default: 512.
        num_mlps (int, optional): The number of MLP layers. Defaults to 8.
        channel_multiplier (int, optional): The mulitiplier factor for the
            channel number. Defaults to 2.
        blur_kernel (list, optional): The blurry kernel. Defaults
            to [1, 3, 3, 1].
        lr_mlp (float, optional): The learning rate for the style mapping
            layer. Defaults to 0.01.
        default_style_mode (str, optional): The default mode of style mixing.
            In training, we defaultly adopt mixing style mode. However, in the
            evaluation, we use 'single' style mode. `['mix', 'single']` are
            currently supported. Defaults to 'mix'.
        eval_style_mode (str, optional): The evaluation mode of style mixing.
            Defaults to 'single'.
        mix_prob (float, optional): Mixing probability. The value should be
            in range of [0, 1]. Defaults to 0.9.
        pretrained (dict | None, optional): Information for pretained models.
            The necessary key is 'ckpt_path'. Besides, you can also provide
            'prefix' to load the generator part from the whole state dict.
            Defaults to None.
        bgr2rgb (bool, optional): Whether to flip the image channel dimension.
            Defaults to False.
    """

    def __init__(self,
                 in_size,
                 out_size,
                 img_channels=3,
                 res_channels=64,
                 num_res=23,
                 style_channels=512,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 pretrained=None,
                 bgr2rgb=False):

        super().__init__()

        # input size must be strictly smaller than output size
        # if in_size >= out_size:
        #     raise ValueError('in_size must be smaller than out_size, but got '
        #                      f'{in_size} and {out_size}.')

        # latent bank (StyleGANv2), with weights being fixed
        self.generator = build_component(
            dict(
                type='StyleGANv2Generator',
                out_size=out_size,
                style_channels=style_channels,
                num_mlps=num_mlps,
                channel_multiplier=channel_multiplier,
                blur_kernel=blur_kernel,
                lr_mlp=lr_mlp,
                default_style_mode=default_style_mode,
                eval_style_mode=eval_style_mode,
                mix_prob=mix_prob,
                pretrained=pretrained,
                bgr2rgb=bgr2rgb))
        self.generator.requires_grad_(False)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                ResFeatureExtractor(img_channels, res_channels, num_blocks=num_res),
                nn.Conv2d(
                    res_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = ResBlock(in_channels, out_channels, blur_kernel)
                
                # nn.Sequential(
                #     nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
                #     nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                #     nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

        # additional modules for StyleGANv2
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))
            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

        # decoder
        decoder_res = [
            2**i
            for i in range(int(np.log2(in_size)), int(np.log2(out_size) + 1))
        ]
        self.decoder = nn.Sequential(
             nn.Conv2d(2 * channels[out_size], 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, 1, 1))
        
        # for res in decoder_res:
        #     if res == in_size:
        #         in_channels = channels[res]
        #     else:
        #         in_channels = 2 * channels[res]

        #     if res < out_size:
        #         out_channels = channels[res * 2]
        #         self.decoder.append(
        #             PixelShufflePack(
        #                 in_channels, out_channels, 2, upsample_kernel=3))
        #     else:
        #         self.decoder.append(
        #             nn.Sequential(
        #                 nn.Conv2d(in_channels, 64, 3, 1, 1),
        #                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                 nn.Conv2d(64, img_channels, 3, 1, 1)))

    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]

        # generator
        injected_noise = [
            getattr(self.generator, f'injected_noise_{i}')
            for i in range(self.generator.num_injected_noises)
        ]
        # 4x4 stage
        out = self.generator.constant_input(latent)
        out = self.generator.conv1(out, latent[:, 0], noise=injected_noise[0])
        skip = self.generator.to_rgb1(out, latent[:, 1])
        _index = 1
        
        # 8x8 ---> higher res
        for up_conv, conv, noise1, noise2, to_rgb in zip(
                self.generator.convs[::2], self.generator.convs[1::2],
                injected_noise[1::2], injected_noise[2::2],
                self.generator.to_rgbs):

            # feature fusion by channel-wise concatenation
            # if out.size(2) <= self.in_size:
            fusion_index = (_index - 1) // 2
            feat = encoder_features[fusion_index]

            out = torch.cat([out, feat], dim=1)
            out = self.fusion_out[fusion_index](out)

            skip = torch.cat([skip, feat], dim=1)
            skip = self.fusion_skip[fusion_index](skip)

            # original StyleGAN operations
            out = up_conv(out, latent[:, _index], noise=noise1)
            out = conv(out, latent[:, _index + 1], noise=noise2)
            skip = to_rgb(out, latent[:, _index + 2], skip)

            # store features for decoder
            # if out.size(2) > self.in_size:
            #     generator_features.append(out)

            _index += 2

        # decoder
        hr = torch.cat([encoder_features[-1], out], dim=1)
        # for i, block in enumerate(self.decoder):
        #     if i > 0:
        #         hr = torch.cat([hr, generator_features[i - 1]], dim=1)
        hr = self.decoder(hr)

        return hr

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

class ResFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=12,
                 blur_kernel=(1,3,3,1),
                 growth_channels=32):

        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(mid_channels, mid_channels, resample_kernel=blur_kernel, downsample=False))
        # self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = nn.Sequential(*layers)
        # self.body = make_layer(
        #     ,
        #     num_blocks,
        #     mid_channels=mid_channels,
        #     growth_channels=growth_channels)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
    
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))
        

class RRDBFeatureExtractor(nn.Module):
    """Feature extractor composed of Residual-in-Residual Dense Blocks (RRDBs).

    It is equivalent to ESRGAN with the upsampling module removed.

    Args:
        in_channels (int): Channel number of inputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64
        num_blocks (int): Block number in the trunk network. Default: 23
        growth_channels (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 num_blocks=23,
                 growth_channels=32):

        super().__init__()

        self.conv_first = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.body = make_layer(
            RRDB,
            num_blocks,
            mid_channels=mid_channels,
            growth_channels=growth_channels, downsample=False)
        self.conv_body = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        feat = self.conv_first(x)
        return feat + self.conv_body(self.body(feat))


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kernel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1), downsample=True):
        super(ResBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(
            in_channels, out_channels, 3, downsample=downsample, resample_kernel=resample_kernel, bias=True, activate=True)
        self.skip = ConvLayer(
            in_channels, out_channels, 1, downsample=downsample, resample_kernel=resample_kernel, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out
    
    
class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size},'
                f' stride={self.stride}, padding={self.padding}, '
                f'bias={self.bias is not None})')
        
def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.

    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.

    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]  # to 2D kernel, outer product
    # normalize
    k /= k.sum()
    return k
        

class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).

    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Default: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor**2)

        if upsample_factor > 1:
            pad = (self.kernel.shape[0] - upsample_factor) - (kernel_size - 1)
            self.pad = ((pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1)
        elif downsample_factor > 1:
            pad = (self.kernel.shape[0] - downsample_factor) + (kernel_size - 1)
            self.pad = ((pad + 1) // 2, pad // 2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}'
                f', downsample_factor={self.downsample_factor})')

class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.

    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)
    
class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kernel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        # downsample
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        # conv
        layers.append(
            EqualConv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias
                and not activate))
        # activation
        if activate:
            if bias:
                layers.append(FusedBiasLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(ConvLayer, self).__init__(*layers)
        
        
