# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Self
from nn_extractor import cfg, nntensor
from nn_extractor.ops.base_op import BaseOp
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
import numpy as np


@dataclass
class Flip(BaseOp):
    '''
    img: the cropped img, represented in np.ndarray.
        in SAR+ coordinate.
    '''
    img: np.ndarray

    '''
    axes: list[int]

    axes, the length need be <= img.ndim
        in SAR+ coordinate. (ndim - 3 as S, ndim - 2 as A, ndim - 1)
    '''
    axes_sar: list[int]

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img = self.img
        axes_sar = self.axes_sar

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'flip.integrate: img is not nntensor: {type(img)}')
            return None

        # ensure region as slice
        if len(axes_sar) > img.ndim:
            cfg.logger.warning(f'flip.integrate: invalid axes: axes: {axes_sar}')
            return None

        return OpItem(
            name=name,
            op_type=OpType.FLIP,
            tensor=img,
            op_params=axes_sar,
        )
