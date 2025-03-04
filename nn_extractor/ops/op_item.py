# -*- coding: utf-8 -*-

import base64
from dataclasses import dataclass
import os
from typing import Any, Optional, Self

from nn_extractor import cfg, nnextractor_pb2, nntensor, utils
from nn_extractor.ops.op_type import OpType
from nn_extractor.recursive_primitive import RecursivePrimitive
from nn_extractor.types import MetaOpItem, NNTensor


@dataclass
class OpItem(object):
    name: str
    op_type: OpType
    tensor: NNTensor
    op_params: Optional[RecursivePrimitive]

    data_id: Optional[str] = None

    def __init__(self: Self, name: str, op_type: OpType, tensor: NNTensor, op_params: Any):
        self.name = name
        self.op_type = op_type
        self.tensor = tensor
        if op_params is None:
            self.op_params = None
        elif isinstance(op_params, RecursivePrimitive):
            self.op_params = op_params
        else:
            self.op_params = RecursivePrimitive(op_params)

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, OpItem):
            return False

        if self.name != other.name:
            return False

        if self.op_type != other.op_type:
            return False

        if not nntensor.eq_nntensor(self.tensor, other.tensor):
            return False

        if self.op_params != other.op_params:
            return False

        return True

    def set_data_id(self, parent_data_id: str):
        self.data_id = f'{parent_data_id}/_op_item_{self.name}'

    def serialize_pb(self) -> nnextractor_pb2.OpItem:
        serialized_tensor = nntensor.serialize_nntensor_pb(self.tensor)

        to_serialize = {
            'name': self.name,
            'op_type': int(self.op_type),
            'tensor': serialized_tensor,
        }
        if self.op_params is not None:
            serialized_op_params = self.op_params.serialize_pb()

            to_serialize['op_params'] = serialized_op_params

        return nnextractor_pb2.OpItem(**to_serialize)

    @classmethod
    def deserialize_pb(cls: Self, op_item_pb: nnextractor_pb2.OpItem) -> Self:
        tensor = nntensor.deserialize_nntensor_pb(op_item_pb.tensor)
        op_params = RecursivePrimitive.deserialize_pb(op_item_pb.op_params)

        return OpItem(
            name=op_item_pb.name,
            op_type=OpType(op_item_pb.op_type),
            tensor=tensor,
            op_params=op_params,
        )

    def save_to_file(self: Self, seq_dir: str):
        serialized = self.serialize_pb()

        filename = f"{self.data_id}.pb"

        out_filename = os.sep.join([cfg.config['output_dir'], seq_dir, filename])
        if os.path.exists(out_filename):
            cfg.logger.warning(f'save_to_file_pb: (item) file already exists: seq_dir: {seq_dir} data_id: {self.data_id} out_filename: {out_filename}')  # noqa

        utils.ensure_dir(out_filename)
        with open(out_filename, 'wb') as f:
            serialized_bytes = serialized.SerializeToString()
            serialized_str = base64.b64encode(serialized_bytes)
            f.write(serialized_str)

    def meta(self: Self):
        meta_tensor = nntensor.meta_nntensor(self.tensor)
        meta_op_params = None if self.op_params is None else self.op_params.meta()
        return MetaOpItem(
            name=self.name,
            op_type=str(self.op_type),
            tensor=meta_tensor,
            op_params=meta_op_params,
        )
