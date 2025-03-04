# -*- coding: utf-8 -*-

from typing import Optional, Self
from enum import Enum
from dataclasses import dataclass

from nn_extractor.items import Items
from nn_extractor.nii import NII
from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
from nn_extractor.recursive_primitive import RecursivePrimitive

from . import nnextractor_pb2

from .types import MetaTaskflow


class TaskflowType(Enum):
    UNSPECIFIED = nnextractor_pb2.TaskflowType.T_UNSPECIFIED

    INPUT = nnextractor_pb2.TaskflowType.T_INPUT

    PREPROCESS = nnextractor_pb2.TaskflowType.T_PREPROCESS

    FORWARD = nnextractor_pb2.TaskflowType.T_FORWARD

    BACKWARD = nnextractor_pb2.TaskflowType.T_BACKWARD

    POSTPROCESS = nnextractor_pb2.TaskflowType.T_POSTPROCESS

    OUTPUT = nnextractor_pb2.TaskflowType.T_OUTPUT

    EXTRACTOR = nnextractor_pb2.TaskflowType.T_EXTRACTOR

    def __int__(self: Self):
        return self.value

    def __str__(self: Self):
        return self.name.lower()


@dataclass
class Taskflow(object):
    the_type: TaskflowType
    flow_id: int
    name: str = ''

    op_type: Optional[OpType] = None
    op_params: Optional[RecursivePrimitive] = None

    def __init__(
        self: Self,
        the_type: TaskflowType,
        flow_id: int,
        name: str,
        items: Optional[Items] = None,
    ):
        self.the_type = the_type
        self.flow_id = flow_id
        self.name = name
        if items is not None:
            for item in items.items:
                if isinstance(item.value, OpItem):
                    self.op_type = item.value.op_type
                    self.op_params = item.value.op_params
                    break
                elif isinstance(item.value, NII):
                    self.op_type = OpType.AFFINE
                    self.op_params = RecursivePrimitive(item.value.affine_ras)

    def serialize(self: Self) -> nnextractor_pb2.Taskflow:
        to_serialize = {
            'the_type': int(self.the_type),
            'flow_id': self.flow_id
        }
        if self.name:
            to_serialize['name'] = self.name

        if self.op_type:
            to_serialize['op_type'] = self.op_type

        if self.op_params:
            serialized_op_params = self.op_params.serialize_pb()
            to_serialize['op_params'] = serialized_op_params

        return nnextractor_pb2.Taskflow(**to_serialize)

    @classmethod
    def deserialize(cls: Self, taskflow_pb2: nnextractor_pb2.Taskflow) -> Self:
        op_params: Optional[RecursivePrimitive] = None
        if taskflow_pb2.op_params:
            op_params = RecursivePrimitive.deserialize_pb(taskflow_pb2.op_params)

        return Taskflow(
            the_type=TaskflowType(taskflow_pb2.the_type),
            flow_id=taskflow_pb2.flow_id,
            name=taskflow_pb2.name,
            op_type=taskflow_pb2.op_type,
            op_params=op_params,
        )

    def meta(self: Self) -> MetaTaskflow:
        meta_op_type = None if self.op_type is None else str(self.op_type)
        meta_op_params = None if self.op_params is None else self.op_params.meta()

        return MetaTaskflow(
            the_type=str(self.the_type),
            flow_id=self.flow_id,
            name=self.name,
            op_type=meta_op_type,
            op_params=meta_op_params,
        )
