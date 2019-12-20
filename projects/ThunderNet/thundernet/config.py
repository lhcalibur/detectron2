# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_thundernet_config(cfg):
    _C = cfg

    _C.MODEL.TRIDENT = CN()

    _C.MODEL.BACKBONE.ARCH = "SNet49"
