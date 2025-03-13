# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Self
from nn_extractor import cfg, nntensor
from nn_extractor.ops.base_op import BaseOp
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
from nn_extractor import utils
from nn_extractor.types import NNTensor


@dataclass
class Crop(BaseOp):
    '''
    img: the cropped img, represented in NNTensor.
        in SAR+ coordinate.
    '''
    img: NNTensor

    '''
    region: list[[start, stop, step]]
        in SAR+ coordinate.

    cropping region, the length need be aligned with img.ndim
    '''
    region_sar: list[int | slice | tuple[Optional[int], Optional[int], Optional[int]]]

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img = self.img
        region_sar = self.region_sar

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'crop.integrate: img is not nntensor: {type(img)}')
            return None

        # ensure region as slice
        sanitized_region_sar = [utils.sanitize_slice(each) for each in region_sar]

        return OpItem(
            name=name,
            op_type=OpType.CROP,
            tensor=img,
            op_params=sanitized_region_sar,
        )
