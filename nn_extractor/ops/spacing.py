# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Self
from nn_extractor import cfg, nntensor
from nn_extractor.ops.base_op import BaseOp
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
from nn_extractor.types import NNTensor, Primitive


@dataclass
class Spacing(BaseOp):
    '''
    img: the padded img, represented in NNTensor
        in SAR+ coordinate.
    '''
    img: NNTensor

    '''
    spacing: Optional[list[Primitive]]
        in RAS+ coordinate.

        if None: align arbitrarily with the reference image.

    spacing, the length need be the same as img.ndim.
    '''
    spacing: Optional[list[Primitive]] = None

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img = self.img
        spacing = self.spacing

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'spacing.integrate: img is not nntensor: {type(img)}')
            return None

        if spacing is None:
            return OpItem(
                name=name,
                op_type=OpType.SPACING,
                tensor=img,
                op_params=None,
            )

        return OpItem(
            name=name,
            op_type=OpType.CROP,
            tensor=img,
            op_params=spacing,
        )
