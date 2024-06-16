from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor
from pcdet.models.detectors.model_mae import MaskedAutoencoderViT
from einops import rearrange
from einops.layers.torch import Rearrange

class BasicBlock(BaseModule):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 dilation: int = 1,
                 downsample: Optional[nn.Module] = None,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None) -> None:
        super(BasicBlock, self).__init__(init_cfg)

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.relu = build_activation_layer(act_cfg)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: Normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: Normalization layer after the second convolution layer.
        """
        return getattr(self, self.norm2_name)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@MODELS.register_module()
class FRNetBackbone(BaseModule):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3))
    }

    def __init__(self,
                 in_channels: int,
                 point_in_channels: int,
                 output_shape: Sequence[int],
                 depth: int,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 point_norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 init_cfg: OptMultiConfig = None,
                 embed_dim=384, 
                 vit_depth=12,
                 num_heads=6,
                 decoder_embed_dim=512,
                 decoder_depth=4,
                 decoder_num_heads=8,
                 mlp_ratio=4,
                 n_clss=20,
                skip_filters=128
                ) -> None:
        super(FRNetBackbone, self).__init__(init_cfg)

        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for FRNetBackbone.')

        self.block, stage_blocks = self.arch_settings[depth]
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        assert len(stage_blocks) == len(out_channels) == len(strides) == len(
            dilations) == num_stages, \
            'The length of stage_blocks, out_channels, strides and ' \
            'dilations should be equal to num_stages.'
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_norm_cfg = point_norm_cfg
        self.act_cfg = act_cfg
        # self.stem = self._make_stem_layer(in_channels, stem_channels)

        self.range_img_size = (64, 512)
        self.patch_size = (2, 8)
        self.GS_H, self.GS_W = self.range_img_size[0] // self.patch_size[0], self.range_img_size[1] // self.patch_size[1]
        self.stem = MaskedAutoencoderViT(
            patch_size=16, embed_dim=embed_dim, depth=vit_depth, num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, norm_layer=nn.LayerNorm, in_chans=5, img_with_size=self.range_img_size, out_chans=5, with_patch_2d=(2, 8), norm_pix_loss=True,
            patch_model='ConvStem', hidden_dim=skip_filters
        )
        self.upconv = UpConvBlock(embed_dim, embed_dim)

        self.point_stem = self._make_point_layer(point_in_channels,
                                                 stem_channels)
        self.fusion_stem = self._make_fusion_layer(stem_channels * 2,
                                                   stem_channels)

        inplanes = stem_channels
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []
        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            dilation = dilations[i]
            planes = out_channels[i]
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.point_fusion_layers.append(
                self._make_point_layer(inplanes + planes, planes))
            self.pixel_fusion_layers.append(
                self._make_fusion_layer(planes * 2, planes))
            self.attention_layers.append(self._make_attention_layer(planes))
            inplanes = planes
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        in_channels = stem_channels + sum(out_channels)
        self.fuse_layers = []
        self.point_fuse_layers = []
        for i, fuse_channel in enumerate(fuse_channels):
            fuse_layer = ConvModule(
                in_channels,
                fuse_channel,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            point_fuse_layer = self._make_point_layer(in_channels,
                                                      fuse_channel)
            in_channels = fuse_channel
            layer_name = f'fuse_layer{i + 1}'
            point_layer_name = f'point_fuse_layer{i + 1}'
            self.add_module(layer_name, fuse_layer)
            self.add_module(point_layer_name, point_fuse_layer)
            self.fuse_layers.append(layer_name)
            self.point_fuse_layers.append(point_layer_name)

    def _make_stem_layer(self, in_channels: int,
                         out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels // 2,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels // 2)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels // 2,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_point_layer(self, in_channels: int,
                          out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            build_norm_layer(self.point_norm_cfg, out_channels)[1],
            nn.ReLU(inplace=True))

    def _make_fusion_layer(self, in_channels: int,
                           out_channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, out_channels)[1],
            build_activation_layer(self.act_cfg))

    def _make_attention_layer(self, channels: int) -> nn.Module:
        return nn.Sequential(
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
            build_conv_layer(
                self.conv_cfg,
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(self.norm_cfg, channels)[1], nn.Sigmoid())

    def make_res_layer(
        self,
        block: nn.Module,
        inplanes: int,
        planes: int,
        num_blocks: int,
        stride: int,
        dilation: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN'),
        act_cfg: ConfigType = dict(type='LeakyReLU')
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes)[1])

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        inplanes = planes
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        return nn.Sequential(*layers)

    def forward_vit_encode(self, x):
        range_latent_full, range_mask_full, range_ids_restore_full, range_skip = self.stem.forward_encoder(x, 0)
        range_latent_full = rearrange(range_latent_full[:, 1:, :], 'b (h w) c -> b c h w', h=self.GS_H, w=self.GS_W) # B, d_model, image_size[0]/patch_stride[0], image_size[1]/patch_stride[1]
        feats = self.upconv(range_latent_full, range_skip)
        return feats

    def forward(self, voxel_dict: dict) -> dict:

        point_feats = voxel_dict['point_feats'][-1] # (N, 256)
        voxel_feats = voxel_dict['voxel_feats']
        voxel_coors = voxel_dict['voxel_coors']
        pts_coors = voxel_dict['coors']
        batch_size = pts_coors[-1, 0].item() + 1

        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        # x = self.stem(x) # (B, 128, 64, 512)
        x = self.forward_vit_encode(x)
        map_point_feats = self.pixel2point(x, pts_coors, stride=1) # (N, C)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        # from IPython import embed; embed()

        point_feats = self.point_stem(fusion_point_feats)
        stride_voxel_coors, frustum_feats = self.point2frustum(
            point_feats, pts_coors, stride=1)
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1)
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        outs = [x]
        out_points = [point_feats]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            # frustum-to-point fusion
            map_point_feats = self.pixel2point(
                x, pts_coors, stride=self.strides[i])
            fusion_point_feats = torch.cat((map_point_feats, point_feats),
                                           dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=self.strides[i])
            pixel_feats = self.frustum2pixel(
                frustum_feats,
                stride_voxel_coors,
                batch_size,
                stride=self.strides[i])
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)
            # residual-attentive
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x
            outs.append(x)
            out_points.append(point_feats)

        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode='bilinear',
                    align_corners=True)

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        for layer_name, point_layer_name in zip(self.fuse_layers,
                                                self.point_fuse_layers):
            fuse_layer = getattr(self, layer_name)
            outs[0] = fuse_layer(outs[0])
            point_fuse_layer = getattr(self, point_layer_name)
            out_points[0] = point_fuse_layer(out_points[0])

        voxel_dict['voxel_feats'] = outs
        voxel_dict['point_feats_backbone'] = out_points
        return voxel_dict

    def frustum2pixel(self,
                      frustum_features: Tensor,
                      coors: Tensor,
                      batch_size: int,
                      stride: int = 1) -> Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype,
            device=frustum_features.device)
        pixel_features[coors[:, 0], coors[:, 1], coors[:,
                                                       2]] = frustum_features
        pixel_features = pixel_features.permute(0, 3, 1, 2).contiguous()
        return pixel_features

    def pixel2point(self,
                    pixel_features: Tensor,
                    coors: Tensor,
                    stride: int = 1) -> Tensor:
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        point_feats = pixel_features[coors[:, 0], coors[:, 1] // stride,
                                     coors[:, 2] // stride]
        return point_feats

    def point2frustum(self,
                      point_features: Tensor,
                      pts_coors: Tensor,
                      stride: int = 1) -> Tuple[Tensor, Tensor]:
        coors = pts_coors.clone()
        coors[:, 1] = pts_coors[:, 1] // stride
        coors[:, 2] = pts_coors[:, 2] // stride
        voxel_coors, inverse_map = torch.unique(
            coors, return_inverse=True, dim=0)
        frustum_features = torch_scatter.scatter_max(
            point_features, inverse_map, dim=0)[0]
        return voxel_coors, frustum_features

class UpConvBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate=0.2,
        scale_factor=(2, 8),
        drop_out=False,
        skip_filters=0):
        super(UpConvBlock, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.skip_filters = skip_filters

        # scale_factor has to be a tuple or a list with two elements
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        assert isinstance(scale_factor, (list, tuple))
        assert len(scale_factor) == 2
        self.scale_factor = scale_factor

        if self.scale_factor[0] != self.scale_factor[1]:
            upsample_layers = [
                nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
                Rearrange('b (c s0 s1) h w -> b c (h s0) (w s1)', s0=self.scale_factor[0], s1=self.scale_factor[1]),]
        else:
            upsample_layers = [
                nn.Conv2d(in_filters, out_filters * self.scale_factor[0] * self.scale_factor[1], kernel_size=(1, 1)),
                nn.PixelShuffle(self.scale_factor[0]),]

        if drop_out:
            upsample_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_upsample = nn.Sequential(*upsample_layers)

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_filters + skip_filters, out_filters, (3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters)
        )
        num_filters = out_filters
        output_layers = [
            nn.Conv2d(num_filters, out_filters, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_filters),
        ]
        if drop_out:
            output_layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_output = nn.Sequential(*output_layers)

    def forward(self, x, skip=None):
        x_up = self.conv_upsample(x) # increase spatial size by a scale factor. B, 2*base_channels, image_size[0], image_size[1]

        if self.skip_filters > 0:
            assert skip is not None
            assert skip.shape[1] == self.skip_filters
            x_up = torch.cat((x_up, skip), dim=1)

        x_up_out = self.conv_output(self.conv1(x_up))
        return x_up_out
