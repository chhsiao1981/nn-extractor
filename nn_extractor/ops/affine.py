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
class Affine(BaseOp):
    '''
    img: the cropped img, represented in NNTensor.
        in SAR+ coordinate.
    '''
    img: NNTensor

    '''
    affine: list[list[number]]
        in RAS+ coordinate.

    affine matrix, 4x4.
    '''
    affine: list[list[Primitive]]

    def integrate(self: Self, name: str) -> Optional[OpItem]:
        img = self.img
        affine = self.affine

        if not nntensor.isinstance_nntensor(img):
            cfg.logger.warning(f'affine.integrate: img is not nntensor: {type(img)}')
            return None

        # ensure region as slice
        sanitized_affine = self._sanitize_affine(affine)
        for each in sanitized_affine:
            if each is None:
                cfg.logger.warning(f'affine.integrate: invalid affine: affine: {affine}')  # noqa
                return None

        return OpItem(
            name=name,
            op_type=OpType.AFFINE,
            tensor=img,
            op_params=sanitized_affine,
        )

    def _sanitize_affine(
        self: Self,
        affine: list[list[Primitive]],
    ) -> Optional[list[list[Primitive]]]:
        if not isinstance(affine, list):
            return None

        if len(affine) != 4:
            return None

        for each in affine:
            if not isinstance(each, list):
                return None
            if len(each) != 4:
                return None

        return affine
