from typing import List

import torch
from detectron2.layers import ShapeSpec
from torch import nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone


class CEM(nn.Module):
    def __init__(self, c3_in_channels, c4_in_channels):
        super().__init__()
        hidden_channels = 245
        self.conv_c3 = nn.Conv2d(c3_in_channels, hidden_channels, 1, 1, 0)
        self.conv_c4 = nn.Conv2d(c4_in_channels, hidden_channels, 1, 1, 0)
        self.fc_glb = nn.Linear(hidden_channels, hidden_channels)
        self.out_channels = 245

    def forward(self, c3, c4):
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        glb = c4.mean([2, 3])
        c4 = F.interpolate(c4, scale_factor=2, mode="nearest", align_corners=None)
        glb = self.fc_glb(glb)
        glb = glb.unsqueeze(-1).unsqueeze(-1)
        x = c3 + c4 + glb
        return x


class SAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = 256
        # for the hidden representation
        self.depthwise_conv_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2,
                                            groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sam_features = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.rpn_in_features_out_channels = out_channels
        self.sam_out_channels = in_channels

        for layer in [self.depthwise_conv_5x5, self.conv_1x1]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, cem):
        x = F.relu(self.depthwise_conv_5x5(cem))
        x = F.relu(self.conv_1x1(x))
        sam_features = self.sam_features(x)
        sam = torch.mul(sam_features, cem)
        return x, sam


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        kernel_size = 5
        rate = 1
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        assert pad_total % 2 == 0
        pad = pad_total // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=kernel_size, stride=self.stride, padding=pad),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=kernel_size, stride=self.stride,
                                padding=pad),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class SNet(Backbone):
    def __init__(self, input_shape: List[ShapeSpec], backbone_arch):
        super(SNet, self).__init__()
        if backbone_arch == "SNet49":
            stages_repeats, stages_out_channels = [3, 7, 3], [24, 60, 120, 240, 512]
        elif backbone_arch == "SNet146":
            stages_repeats, stages_out_channels = [3, 7, 3], [24, 132, 264, 528]
        elif backbone_arch == "SNet535":
            stages_repeats, stages_out_channels = [3, 7, 3], [48, 248, 496, 992]
        else:
            raise RuntimeError(f"BackBone arch: {backbone_arch} not valid")

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 4 and len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 4 or 5 positive ints')
        self._stage_out_channels = stages_out_channels
        self.include_conv5 = len(stages_out_channels) == 5

        self._out_feature_strides = {}
        self._out_feature_channels = {}

        input_channels = input_shape.channels
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            if name == 'stage4' and self.include_conv5:
                input_channels = output_channels
                output_channels = self._stage_out_channels[-1]
                seq.append(
                    nn.Sequential(
                        nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=True),
                    )
                )
            setattr(self, name, nn.Sequential(*seq))
            self._out_feature_channels[name] = output_channels
            input_channels = output_channels
        self.cem = CEM(self._out_feature_channels['stage3'], self._out_feature_channels['stage4'])
        self.sam = SAM(self.cem.out_channels)

        self._out_features = ['stage3', 'stage4', 'cem', 'sam', 'rpn_in_features']
        self._out_feature_strides['stage3'] = 16
        self._out_feature_strides['stage4'] = 32
        self._out_feature_strides['cem'] = 16
        self._out_feature_strides['sam'] = 16
        self._out_feature_strides['rpn_in_features'] = 16
        self._out_feature_channels['cem'] = self.cem.out_channels
        self._out_feature_channels['sam'] = self.sam.sam_out_channels
        self._out_feature_channels['rpn_in_features'] = self.sam.rpn_in_features_out_channels

    def forward(self, x):
        outputs = {}
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        outputs['stage3'] = x
        x = self.stage4(x)
        outputs['stage4'] = x
        cem = self.cem(outputs['stage3'], outputs['stage4'])
        outputs['cem'] = cem
        rpn_in_features, sam = self.sam(cem)
        outputs['rpn_in_features'] = rpn_in_features
        outputs['sam'] = sam
        return outputs

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return self._out_feature_strides['stage4']


@BACKBONE_REGISTRY.register()
class SNet49(SNet):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(input_shape, 'SNet49')
