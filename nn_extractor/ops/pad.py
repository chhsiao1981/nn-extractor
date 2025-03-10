# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Self
from nn_extractor import cfg, nntensor
from nn_extractor.ops import utils
from nn_extractor.ops.base_op import BaseOp
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
from nn_extractor.types import NNTensor


@dataclass
class Pad(BaseOp):
    '''
    img: the padded img, represented in NNTensor
        in SAR+ coordinate.
    '''
    img: NNTensor

    '''
    slicer_revert_padding: list[[start, stop, step]]
        in SAR+ coordinate.

    reverse padding (as cropping, img[slicer_revert_padding] to pre-padding img),
    the length need be aligned with img.ndim.
    '''
    slicer_revert_padding: list[int | slice | tuple[Optional[int], Optional[int], Optional[int]]]  # noqa

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img: NNTensor = self.img
        slicer_revert_padding = self.slicer_revert_padding

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'pad.integrate: img is not nntensor: {type(img)}')
            return None

        # ensure region as slice
        sanitized_slicer_revert_padding = [
            utils.sanitize_slice(each)
            for each in slicer_revert_padding
        ]

        return OpItem(
            name=name,
            op_type=OpType.PAD,
            tensor=img,
            op_params=sanitized_slicer_revert_padding,
        )
