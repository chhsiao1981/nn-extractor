# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional, Self
from nn_extractor import cfg
from nn_extractor import nntensor
from nn_extractor.ops.base_op import BaseOp
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
from nn_extractor.types import NNTensor, Primitive


@dataclass
class Direction(BaseOp):
    '''
    img: the cropped img, represented in NNTensor.
        in SAR+ coordinate.
    '''
    img: NNTensor

    '''
    direction: list[list[number]]
        in RAS+ coordinate.

    cropping region, the length need be aligned with img.ndim
    '''
    direction: list[list[Primitive]]

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img = self.img
        direction = self.direction

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'direction.integrate: img is not nntensor: {type(img)}')
            return None

        # ensure region as slice
        sanitized_direction = self._sanitize_direction(direction)
        if sanitized_direction is None:
            cfg.logger.warning(f'direction.integrate: invalid direction: direction: {direction}')  # noqa
            return None

        return OpItem(
            name=name,
            op_type=OpType.DIRECTION,
            tensor=img,
            op_params=sanitized_direction,
        )

    def _sanitize_direction(
        self: Self,
        direction: list[list[Primitive]],
    ) -> Optional[list[list[Primitive]]]:
        if not isinstance(direction, list):
            return None

        if len(direction) != 3:
            return None

        for each in direction:
            if not isinstance(each, list):
                return None
            if len(each) != 3:
                return None

        return direction
