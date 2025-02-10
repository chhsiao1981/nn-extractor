# -*- coding: utf-8 -*-

from typing import Self, Any, Optional, TypedDict
from dataclasses import dataclass

import itertools

from . import nntensor
from . import nnextractor_pb2
from .types import MetaNII, NNTensor


@dataclass
class NII(object):
    tensor: NNTensor
    origin_ras: list[float]
    direction_ras: list[list[float]]
    spacing_ras: list[float]
    affine_ras: Optional[list[list[float]]] = None

    def __eq__(self: Self, value: Any) -> bool:
        if not isinstance(value, NII):
            return False
        if not nntensor.eq_nntensor(self.tensor, value.tensor):
            return False
        if not self.origin_ras == value.origin_ras:
            return False
        if not self.direction_ras == value.direction_ras:
            return False
        if not self.spacing_ras == value.spacing_ras:
            return False
        if not self.affine_ras == value.affine_ras:
            return False
        return True

    def serialize_pb(self: Self) -> nnextractor_pb2.NII:
        serialized_tensor = nntensor.serialize_nntensor_pb(self.tensor)
        serialized_direction = list(itertools.chain.from_iterable(self.direction_ras))
        affine = [] if self.affine_ras is None \
            else list(itertools.chain.from_iterable(self.affine_ras))

        return nnextractor_pb2.NII(
            tensor=serialized_tensor,
            origin=self.origin_ras,
            direction=serialized_direction,
            spacing=self.spacing_ras,
            affine=affine)

    @classmethod
    def deserialize_pb(cls: Self, nii_pb: nnextractor_pb2.NII) -> Self:
        the_tensor = nntensor.deserialize_nntensor_pb(nii_pb.ndarray)
        origin_ras = nii_pb.origin
        direction_ras = [direction_1d_to_2d(nii_pb.direction)]
        spacing_ras = nii_pb.spacing
        affine_ras = None if not nii_pb.direction else [affine_1d_to_2d(nii_pb.direction)]

        return NII(
            tensor=the_tensor,
            origin_ras=origin_ras,
            direction_ras=direction_ras,
            spacing_ras=spacing_ras,
            affine_ras=affine_ras,
        )

    def meta(self: Self) -> MetaNII:
        meta_tensor = nntensor.meta_nntensor(self.tensor)
        return MetaNII(
            tensor=meta_tensor,
            origin_ras=self.origin_ras,
            direction_ras=self.direction_ras,
            spacing_ras=self.spacing_ras,
            affine_ras=self.affine_ras,
        )


class SITKProps(TypedDict):
    spacing: list[float]
    origin: list[float]
    direction: list[float]


class SimpleITKIOProps(TypedDict):
    sitk_stuff: SITKProps
    spacing: list[float]


def from_sitk_image_props(image: NNTensor, props: SimpleITKIOProps) -> NII:
    sitk_props = props['sitk_stuff']
    spacing = sitk_props['spacing']
    origin_lps = sitk_props['origin']
    direction_lps_1d = sitk_props['direction']
    direction_lps = direction_1d_to_2d(direction_lps_1d)
    affine_lps = [
        [direction_lps[0][0] * spacing[0], direction_lps[0][1], direction_lps[0][2], origin_lps[0]],
        [direction_lps[1][0], direction_lps[1][1] * spacing[1], direction_lps[1][2], origin_lps[1]],
        [direction_lps[2][0], direction_lps[2][1], direction_lps[2][2] * spacing[2], origin_lps[2]],
        [0, 0, 0, 1],
    ]

    return NII(
        tensor=image,
        origin_ras=[-origin_lps[0], -origin_lps[1], origin_lps[2]],
        direction_ras=[
            [-direction_lps[0][0], -direction_lps[0][1], -direction_lps[0][2]],
            [-direction_lps[1][0], -direction_lps[1][1], -direction_lps[1][2]],
            [direction_lps[2][0], direction_lps[2][1], direction_lps[2][2]],
        ],
        spacing_ras=spacing,
        affine_ras=[
            [-affine_lps[0][0], -affine_lps[0][1], -affine_lps[0][2], -affine_lps[0][3]],
            [-affine_lps[1][0], -affine_lps[1][1], -affine_lps[1][2], -affine_lps[1][3]],
            [affine_lps[2][0], affine_lps[2][1], affine_lps[2][2], affine_lps[2][3]],
            [affine_lps[3][0], affine_lps[3][1], affine_lps[3][2], affine_lps[3][3]],
        ]
    )


def affine_1d_to_2d(affine_1d: list[float]) -> list[list[float]]:
    return [affine_1d[(idx * 4):(idx * 4 + 4)] for idx in range(4)]


def direction_1d_to_2d(direction_1d: list[float]) -> list[list[float]]:
    return [direction_1d[(idx * 3):(idx * 3 + 3)] for idx in range(3)]
