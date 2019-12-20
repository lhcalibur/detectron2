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
        self.fc_glb = nn.Linear(c4_in_channels, hidden_channels)
        self.out_channels = 245

    def forward(self, c3, c4):
        c3 = self.conv_c3(c3)
        c4 = self.conv_c4(c4)
        glb = c4.mean([2, 3])
        c4 = F.interpolate(c4, scale_factor=2, mode="nearest", align_corners=False)
        glb = self.fc_glb(glb)
        x = c3 + c4 + glb
        return x


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

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
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
            self.depthwise_conv(branch_features, branch_features, kernel_size=5, stride=self.stride, padding=2),
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


@BACKBONE_REGISTRY.register()
class SNet(Backbone):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(SNet, self).__init__()
        backbone_arch = cfg.MODEL.BACKBONE.ARCH
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
        self.cem = CEM(self._stage_out_channels[-2], self._stage_out_channels[-1])

        self._out_features = ['stage3', 'stage4', 'cem']
        self._out_feature_strides['stage3'] = 16
        self._out_feature_strides['stage4'] = 32
        self._out_feature_strides['cem'] = 16
        self._out_feature_channels['cem'] = self.cem.out_channels

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
        return outputs
