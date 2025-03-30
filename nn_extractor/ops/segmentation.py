# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Optional, Self

from nn_extractor import cfg, nntensor, utils
from nn_extractor.ops.base_op import BaseOp
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
from nn_extractor.types import NNTensor


@dataclass
class Segmentation(BaseOp):
    '''
    img: the cropped img, represented in NNTensor.
        in SAR+ coordinate.
    '''
    img: NNTensor

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img = self.img

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'Segmentation.integrate: img is not nntensor: {type(img)}')
            return None

        return OpItem(
            name=name,
            op_type=OpType.SEGMENTATION,
            tensor=img,
        )
