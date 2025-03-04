# -*- coding: utf-8 -*-

from typing import TypedDict, Optional, Self

from nn_extractor.ops.op_type import OpType
import numpy as np
import torch

type Primitive = None | bool | str | float | int | np.bool | np.float16 | np.float64 | np.float32 | np.int8 | np.int16 | np.int64 | np.int32  # noqa

type NNTensor = np.ndarray | torch.Tensor

type RecursiveNNTensor = NNTensor | list[RecursiveNNTensor] | dict[str, RecursiveNNTensor]

type MetaPrimitive = Primitive

type MetaRecursivePrimitive = Primitive | list[MetaRecursivePrimitive] | dict[str, MetaRecursivePrimitive]  # noqa


class MetaNNTensor(TypedDict):
    shape: list[int]
    the_type: str


class MetaNII(TypedDict):
    tensor: MetaNNTensor
    origin_ras: list[float]
    direction_ras: list[list[float]]
    spacing_ras: list[float]
    affine_ras: Optional[list[list[float]]]


type RecursiveMetaItemValue = MetaPrimitive | MetaNNTensor | MetaNII | list[RecursiveMetaItemValue] | dict[str, RecursiveMetaItemValue]  # noqa


class MetaItem(TypedDict):
    name: str
    the_type: str
    value: RecursiveMetaItemValue | list[Self] | dict[str, Self]
    data_id: str


class MetaOpItem(TypedDict):
    name: str
    op_type: str
    tensor: MetaNNTensor
    op_params: MetaRecursivePrimitive


class MetaNNRecordMeta(TypedDict):
    shape: list[int]
    the_type: str
    is_meta_only: bool


class MetaNNRecord(TypedDict):
    name: str
    the_type: str
    record: MetaNNTensor | MetaNNRecordMeta | list[Self] | dict[str, Self]
    data_id: str


class MetaNNParameter(TypedDict):
    name: str
    parameter: MetaNNRecord


class ProfileValue(TypedDict):
    start: int
    diff: int
    last_diff: int
    count: int


class MetaNNNode(TypedDict):
    name: str

    inputs: list[MetaNNRecord]
    params: list[MetaNNParameter]
    activation: Optional[MetaNNRecord]

    gradient_inputs: list[MetaNNRecord]
    gradient_params: list[MetaNNParameter]
    gradients: list[MetaNNRecord]

    children: Optional[list[Self]]

    the_dir: str
    filename: str


class MetaItems(TypedDict):
    name: str
    items: list[MetaItem]


class MetaTaskflow(TypedDict):
    the_type: str
    flow_id: int
    name: str = ''
    op_type: Optional[OpType] = None
    op_params: Optional[MetaRecursivePrimitive] = None


class MetaNNExtractor(TypedDict):
    name: str

    taskflow: list[MetaTaskflow]
    inputs: list[MetaItems]
    preprocesses: list[MetaItems]
    nodes: list[MetaNNNode]
    postprocesses: list[MetaItems]
    outputs: list[MetaItems]
    extractors: list[Self]

    parent_name: str
