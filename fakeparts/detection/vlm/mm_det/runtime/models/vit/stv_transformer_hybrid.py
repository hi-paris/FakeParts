""" Hybrid Vision Transformer (ViT) in PyTorch

A PyTorch implement of the Hybrid Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

NOTE These hybrid model definitions depend on code in vision_transformer.py.
They were moved here to keep file sizes sane.

Hacked together by / Copyright 2020, Ross Wightman
"""
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import StdConv2dSame, StdConv2d, to_2tuple, Format, nchw_to
from ._registry import generate_default_cfgs, register_model, register_model_deprecations
from .resnetv2 import ResNetV2
from .vision_transformer import VisionTransformer, _create_vision_transformer, _create_window_vision_transformer_with_iafa


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            backbone,
            img_size=224,
            patch_size=1,
            feature_size=None,
            in_chans=3,
            embed_dim=768,
            bias=True,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                    
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        if not dynamic_img_pad:
            assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        _, _, H, W = x.shape
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        return x


class AdaptiveSequenceHybridEmbedForRecons(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map sequence from original and reconstruction frames by using CNN, then fuse feature maps, flatten, project to embedding dim.
    Mutiple the patch number by window size.
    """
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            rgb_backbone,
            recons_backbone,
            img_size=224,
            patch_size=1,
            feature_size=None,
            in_chans=3,
            embed_dim=768,
            bias=True,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
            window_size = 10
    ):
        super().__init__()
        assert isinstance(rgb_backbone, nn.Module)
        assert isinstance(recons_backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.rgb_backbone = rgb_backbone
        self.recons_backbone = recons_backbone
        self.window_size = window_size
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = rgb_backbone.training
                if training:
                    rgb_backbone.eval()
                
                o = self.rgb_backbone(torch.zeros(self.window_size, in_chans, img_size[0], img_size[1]))    # add window_size into the input
                # BL, FC, FH, FW = o.shape
                
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                    
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                rgb_backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.rgb_backbone, 'feature_info'):
                feature_dim = self.rgb_backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.rgb_backbone.num_features
        if not dynamic_img_pad:
            assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.window_size    # increase patch size by window size
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.transition = nn.Sequential(
            nn.Conv2d(2 * feature_dim, feature_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        
        customized_components = [self.transition]
        for component in customized_components:
            for m in component.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)

    def forward(self, x_input):
        x_original, x_recons = x_input
        B = x_original.shape[0]
        x_original = rearrange(x_original, 'b l c h w -> (b l) c h w')
        x_recons = rearrange(x_recons, 'b l c h w -> (b l) c h w')
        feat_rgb = self.rgb_backbone(x_original)
        # x = feat_rgb
        feat_recons = self.recons_backbone(x_recons)
        x = torch.cat([feat_rgb, feat_recons], dim=1)
        # BL, FC, FH, FW = x.shape
        
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        _, _, H, W = x.shape
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.transition(x)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = rearrange(x, '(b l) m d -> b (l m) d', b=B)
        return x
    

def _create_vision_transformer_hybrid(variant, backbone, pretrained=False, **kwargs):
    embed_layer = partial(HybridEmbed, backbone=backbone)
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_vision_transformer(variant, pretrained=pretrained, embed_layer=embed_layer, **kwargs)


def _create_window_vision_transformer_hybrid_with_recons_iafa(variant, rgb_backbone, recons_backbone, window_size, pretrained=False, **kwargs):
    embed_layer = partial(AdaptiveSequenceHybridEmbedForRecons, rgb_backbone=rgb_backbone, recons_backbone=recons_backbone, window_size=window_size)    # add window size for patch embed
    kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
    return _create_window_vision_transformer_with_iafa(variant, pretrained=pretrained, embed_layer=embed_layer, **kwargs)


def _resnetv2(layers=(3, 4, 9), **kwargs):
    """ ResNet-V2 backbone helper"""
    padding_same = kwargs.get('padding_same', True)
    stem_type = 'same' if padding_same else ''
    conv_layer = partial(StdConv2dSame, eps=1e-8) if padding_same else partial(StdConv2d, eps=1e-8)
    if len(layers):
        backbone = ResNetV2(
            layers=layers, num_classes=0, global_pool='', in_chans=kwargs.get('in_chans', 3),
            preact=False, stem_type=stem_type, conv_layer=conv_layer)
    else:
        raise NotImplementedError    # should never reach here
        # backbone = create_resnetv2_stem(
        #     kwargs.get('in_chans', 3), stem_type=stem_type, preact=False, conv_layer=conv_layer)
    return backbone


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.backbone.stem.conv', 'classifier': 'head',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # hybrid in-1k models (weights from official JAX impl where they exist)
    'vit_tiny_r_s16_p8_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True,
        first_conv='patch_embed.backbone.conv'),
    'vit_tiny_r_s16_p8_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        first_conv='patch_embed.backbone.conv', input_size=(3, 384, 384), crop_pct=1.0, custom_load=True),
    'vit_small_r26_s32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.03-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True,
    ),
    'vit_small_r26_s32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0, custom_load=True),
    'vit_base_r26_s32_224.untrained': _cfg(),
    'vit_base_r50_s16_384.orig_in21k_ft_in1k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_r50_s32_224.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',
        hf_hub_id='timm/',
        custom_load=True,
    ),
    'vit_large_r50_s32_384.augreg_in21k_ft_in1k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        hf_hub_id='timm/',
        input_size=(3, 384, 384), crop_pct=1.0, custom_load=True,
    ),

    # hybrid in-21k models (weights from official Google JAX impl where they exist)
    'vit_tiny_r_s16_p8_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R_Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9, first_conv='patch_embed.backbone.conv', custom_load=True),
    'vit_small_r26_s32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R26_S_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.03-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9, custom_load=True),
    'vit_base_r50_s16_224.orig_in21k': _cfg(
        # url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        url = '/home/sxf/conn/R50_L_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz',    # it is the augreg_in21k_ft_in1k weights
        hf_hub_id='timm/',
        num_classes=0, crop_pct=0.9),
    'vit_large_r50_s32_224.augreg_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/R50_L_32-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz',
        hf_hub_id='timm/',
        num_classes=21843, crop_pct=0.9, custom_load=True),

    # hybrid models (using timm resnet backbones)
    'vit_small_resnet26d_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
    'vit_small_resnet50d_s16_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
    'vit_base_resnet26d_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
    'vit_base_resnet50d_224.untrained': _cfg(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, first_conv='patch_embed.backbone.conv1.0'),
})


@register_model
def vit_tiny_r_s16_p8_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 224 x 224.
    """
    backbone = _resnetv2(layers=(), **kwargs)
    model_args = dict(patch_size=8, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_r_s16_p8_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_tiny_r_s16_p8_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R+ViT-Ti/S16 w/ 8x8 patch hybrid @ 384 x 384.
    """
    backbone = _resnetv2(layers=(), **kwargs)
    model_args = dict(patch_size=8, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer_hybrid(
        'vit_tiny_r_s16_p8_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_r26_s32_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R26+ViT-S/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_hybrid(
        'vit_small_r26_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_r26_s32_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R26+ViT-S/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer_hybrid(
        'vit_small_r26_s32_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r26_s32_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R26+ViT-B/S32 hybrid.
    """
    backbone = _resnetv2((2, 2, 2, 2), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r26_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r50_s16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-B/S16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_r50_s16_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-B/16 hybrid from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_r50_s16_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_r50_s32_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-L/S32 hybrid.
    """
    backbone = _resnetv2((3, 4, 6, 3), **kwargs)
    model_args = dict(embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer_hybrid(
        'vit_large_r50_s32_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_large_r50_s32_384(pretrained=False, **kwargs) -> VisionTransformer:
    """ R50+ViT-L/S32 hybrid.
    """
    backbone = _resnetv2((3, 4, 6, 3), **kwargs)
    model_args = dict(embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer_hybrid(
        'vit_large_r50_s32_384', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_resnet26d_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT small hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3)
    model = _create_vision_transformer_hybrid(
        'vit_small_resnet26d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_small_resnet50d_s16_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT small hybrid w/ ResNet50D 3-stages, stride 16. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[3])
    model_args = dict(embed_dim=768, depth=8, num_heads=8, mlp_ratio=3)
    model = _create_vision_transformer_hybrid(
        'vit_small_resnet50d_s16_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_resnet26d_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT base hybrid w/ ResNet26D stride 32. No pretrained weights.
    """
    backbone = resnet26d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_resnet26d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_resnet50d_224(pretrained=False, **kwargs) -> VisionTransformer:
    """ Custom ViT base hybrid w/ ResNet50D stride 32. No pretrained weights.
    """
    backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    model_args = dict(embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_hybrid(
        'vit_base_resnet50d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    return model


register_model_deprecations(__name__, {
    'vit_tiny_r_s16_p8_224_in21k': 'vit_tiny_r_s16_p8_224.augreg_in21k',
    'vit_small_r26_s32_224_in21k': 'vit_small_r26_s32_224.augreg_in21k',
    'vit_base_r50_s16_224_in21k': 'vit_base_r50_s16_224.orig_in21k',
    'vit_base_resnet50_224_in21k': 'vit_base_r50_s16_224.orig_in21k',
    'vit_large_r50_s32_224_in21k': 'vit_large_r50_s32_224.augreg_in21k',
    'vit_base_resnet50_384': 'vit_base_r50_s16_384.orig_in21k_ft_in1k'
})


def adaptive_load(model, state_dict, in_chans):
    # for vit base
    pretrained_cfg = {
        'first_conv': 'patch_embed.backbone.stem.conv',
        'classifier': 'head',
        'label_offset': 0,
        'num_classes': 0
    }
    num_classes = 0
    strict = False
    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                from timm.models._manipulate import adapt_input_conv
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                print(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                print(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]
    
    load_result = model.load_state_dict(state_dict, strict=strict)
    if load_result.missing_keys:
        print(
            f'Missing keys ({", ".join(load_result.missing_keys)}) discovered while loading pretrained weights.'
            f' This is expected if model is being adapted.')
    if load_result.unexpected_keys:
        print(
            f'Unexpected keys ({", ".join(load_result.unexpected_keys)}) found while loading pretrained weights.'
            f' This may be expected if model is being adapted.')


@register_model
def vit_base_r50_s16_224_with_recons_iafa(pretrained=False, ckpt_path=None, in_chans=3, window_size=10, **kwargs):
    """ R50+ViT-B/S16 hybrid + residual branch, with reconstruction and In-and-Across Frame Attention
    """
    rgb_backbone = _resnetv2((3, 4, 9), **kwargs)
    recons_backbone = _resnetv2((3, 4, 9), **kwargs)
    model_args = dict(in_chans=in_chans, embed_dim=768, depth=12, num_heads=12)    # add in_chans as paraemter
    
    if ckpt_path is None:
        model = _create_window_vision_transformer_hybrid_with_recons_iafa(
        'vit_base_r50_s16_224', 
        rgb_backbone=rgb_backbone, 
        recons_backbone=recons_backbone, 
        pretrained=pretrained, 
        window_size=window_size, 
        **dict(model_args, **kwargs)
    )
    else:
        # load from local file, pretrained option is disabled.
        model = _create_window_vision_transformer_hybrid_with_recons_iafa(
        'vit_base_r50_s16_224', 
        rgb_backbone=rgb_backbone, 
        recons_backbone=recons_backbone, 
        pretrained=False, 
        window_size=window_size, 
        **dict(model_args, **kwargs)
    )
        from timm.models._helpers import load_state_dict
        state_dict = load_state_dict(ckpt_path)
        adaptive_load(model, state_dict, in_chans=in_chans)
        
        backbone_state_dict = dict()
        for key, value in state_dict.items():
            if 'patch_embed.backbone' in key:
                rgb_weight_key = key.replace("patch_embed.backbone.","patch_embed.rgb_backbone")
                recons_weight_key = key.replace("patch_embed.backbone.","patch_embed.recons_backbone")
                backbone_state_dict[rgb_weight_key] = value
                backbone_state_dict[recons_weight_key] = value
            if 'blocks' in key and 'patch_embed' not in key:
                if 'attn' in key:
                    weight_key = key.replace('attn', 'temporal_attn')
                    backbone_state_dict[weight_key] = value
                if 'mlp' in key:
                    weight_key = key.replace('mlp', 'temporal_mlp')
                    backbone_state_dict[weight_key] = value
                if 'norm' in key:
                    weight_key = key.replace('norm', 'temporal_norm')
                    backbone_state_dict[weight_key] = value
        model.load_state_dict(backbone_state_dict, strict=False)
        print(f'Load pretrained Hybrid ViT weights from {ckpt_path}')

    # copy pretrained weights for cnn custom modules
    rgb_stem_weights =  model.patch_embed.rgb_backbone.stem.state_dict()
    model.patch_embed.rgb_backbone.freq_stem.load_state_dict(rgb_stem_weights)
    recons_stem_weights =  model.patch_embed.recons_backbone.stem.state_dict()
    model.patch_embed.recons_backbone.freq_stem.load_state_dict(recons_stem_weights)
    
    # copy pretrained pos embed and extend to window size
    cls_pos_embed = model.pos_embed[:, 0, :]
    cls_token = model.cls_token
    for i in range(window_size):
        model.temporal_pos_embed.data[:, i, :] = cls_pos_embed.data
        model.fc_token.data[:, i, :] = cls_token.data

    return model