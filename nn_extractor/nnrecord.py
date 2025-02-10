# -*- coding: utf-8 -*-

from enum import Enum
from dataclasses import dataclass
from typing import Self, Any, Optional

import os

import base64

from nn_extractor.ops.op_item import OpItem

from . import cfg
from . import nnextractor_pb2
from . import nntensor

from .types import MetaNNRecord, NNTensor, RecursiveNNTensor, MetaNNRecordMeta
from . import utils


@dataclass
class NNRecordMeta(object):
    shape: list[int]
    the_type: nntensor.NNTensorType

    def meta(self: Self) -> MetaNNRecordMeta:
        return MetaNNRecordMeta(shape=self.shape, the_type=str(self.the_type), is_meta_only=True)

    def __eq__(self: Self, value: Any) -> bool:
        if not isinstance(value, NNRecordMeta):
            return False
        return self.shape == value.shape and self.the_type == value.the_type


type RecursiveNNRecordMeta = NNRecordMeta | list[RecursiveNNRecordMeta] | dict[str, RecursiveNNRecordMeta]  # noqa


class NNRecordType(Enum):
    UNSPECIFIED = nnextractor_pb2.NNRecordType.NNR_UNSPECIFIED
    ARRAY = nnextractor_pb2.NNRecordType.NNR_ARRAY
    LIST = nnextractor_pb2.NNRecordType.NNR_LIST
    MAP = nnextractor_pb2.NNRecordType.NNR_MAP
    META = nnextractor_pb2.NNRecordType.NNR_META
    OP_ITEM = nnextractor_pb2.NNRecordType.NNR_OP_ITEM

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return self.name.lower()


@dataclass
class NNRecord(object):
    name: str

    value: NNTensor | list[Self] | dict[str, Self] | NNRecordMeta | OpItem

    the_type: NNRecordType = NNRecordType.UNSPECIFIED

    data_id: Optional[str] = None

    def __init__(
        self: Self,

        value: RecursiveNNTensor | NNRecordMeta | list[Self] | dict[str, Self] | OpItem,

        name: str = '',

        the_type: Optional[NNRecordType] = None,

        data_id: Optional[str] = None,
    ):
        '''
        NNRecord

        It is usual that we have only the value and the name is already determined by nnnode,
        making the name as extremely unnecessary variable.
        We put value as the 1st parameter.
        '''

        if the_type is None:
            if nntensor.isinstance_nntensor(value):
                the_type = NNRecordType.ARRAY
            elif isinstance(value, list) or isinstance(value, tuple):
                the_type = NNRecordType.LIST
            elif isinstance(value, dict):
                the_type = NNRecordType.MAP
            elif isinstance(value, NNRecordMeta):
                the_type = NNRecordType.META
            elif isinstance(value, OpItem):
                the_type = NNRecordType.OP_ITEM
            else:
                raise Exception(
                    f'NNRecord.__init__: invalid type: name: {name} type: {type(value)}')
        if the_type == NNRecordType.LIST:
            value = [NNRecord._single_nnrecord(each, data_id, idx)
                     for idx, each in enumerate(value)]
        elif the_type == NNRecordType.MAP:
            value = {k: NNRecord._single_nnrecord(each, data_id, k) for k, each in value.items()}
        elif the_type == NNRecordType.OP_ITEM:
            value.set_data_id(data_id)

        self.name = name
        self.the_type = the_type
        self.value = value
        self.data_id = data_id

    @classmethod
    def _single_nnrecord(cls: Self, value: Any, data_id: Optional[str], the_idx: int | str) -> Self:
        if isinstance(value, NNRecord):
            return value

        child_data_id = None if data_id is None else f'{data_id}/{the_idx}'

        return NNRecord(value, data_id=child_data_id)

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, NNRecord):
            return False
        if self.name != other.name:
            return False
        if self.the_type != other.the_type:
            return False

        if self.the_type == NNRecordType.ARRAY:
            return nntensor.eq_nntensor(self.value, other.value)
        else:
            return self.value == other.value

    def serialize_pb(self: Self) -> nnextractor_pb2.NNRecord:
        if self.the_type == NNRecordType.ARRAY:
            serialized_tensor = nntensor.serialize_nntensor_pb(self.value)
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), tensor=serialized_tensor)
        elif self.the_type == NNRecordType.LIST:
            serialized_list = [each.serialize_pb() for each in self.value]
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), records=serialized_list)
        elif self.the_type == NNRecordType.MAP:
            serialized_dict = {k: each.serialize_pb() for k, each in self.value.items()}
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), the_map=serialized_dict)
        elif self.the_type == NNRecordType.META:
            serialized_type = int(self.value.the_type)
            serialized_meta = nnextractor_pb2.NNRecordMeta(
                shape=self.value.shape,
                the_type=serialized_type,
            )
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), meta=serialized_meta)
        elif self.the_type == NNRecordType.OP_ITEM:
            serialized_value = self.value.serialize_pb()
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), op_item=serialized_value)
        else:
            raise Exception(f'NNRecord.serialize: invalid type: {self.the_type}')

    @classmethod
    def deserialize_pb(cls: Self, nnrecord_pb: nnextractor_pb2.NNRecord) -> Self:
        if nnrecord_pb.the_type == nnextractor_pb2.NNRecordType.NNR_ARRAY:
            the_array = nntensor.deserialize_nntensor_pb(nnrecord_pb.tensor)
            return NNRecord(name=nnrecord_pb.name, value=the_array, the_type=NNRecordType.ARRAY)
        elif nnrecord_pb.the_type == nnextractor_pb2.NNRecordType.NNR_LIST:
            the_list = [cls.deserialize_pb(each) for each in nnrecord_pb.records]
            return NNRecord(name=nnrecord_pb.name, value=the_list, the_type=NNRecordType.LIST)
        elif nnrecord_pb.the_type == nnextractor_pb2.NNRecordType.NNR_MAP:
            the_dict = {k: cls.deserialize_pb(each) for k, each in nnrecord_pb.the_map.items()}
            return NNRecord(name=nnrecord_pb.name, value=the_dict, the_type=NNRecordType.MAP)
        elif nnrecord_pb.the_type == nnextractor_pb2.NNRecordType.NNR_META:
            meta_type = nntensor.NNTensorType(nnrecord_pb.meta.the_type)
            the_meta = NNRecordMeta(shape=nnrecord_pb.meta.shape, the_type=meta_type)
            return NNRecord(name=nnrecord_pb.name, value=the_meta, the_type=NNRecordType.META)
        elif nnrecord_pb.the_type == nnextractor_pb2.NNRecordType.NNR_OP_ITEM:
            value = OpItem.deserialize_pb(nnrecord_pb.op_item)
            return NNRecord(name=nnrecord_pb.name, value=value, the_type=NNRecordType.OP_ITEM)
        else:
            raise Exception(f'NNRecord.deserialize: invalid type: {nnrecord_pb.the_type}')

    def meta(self: Self) -> MetaNNRecord:
        ret: MetaNNRecord = None
        if self.the_type == NNRecordType.ARRAY:
            ret = nntensor.meta_nntensor(self.value)
        elif self.the_type == NNRecordType.LIST:
            ret = [each.meta() for each in self.value]
        elif self.the_type == NNRecordType.MAP:
            ret = {k: each.meta() for k, each in self.value.items()}
        elif self.the_type == NNRecordType.META:
            ret = self.value.meta()
        elif self.the_type == NNRecordType.OP_ITEM:
            ret = self.value.meta()
        else:
            raise Exception(f'NNRecord.meta: invalid type: {self.the_type}')

        return MetaNNRecord(
            name=self.name,
            record=ret,
            data_id=self.data_id,
            the_type=str(self.the_type),
        )

    def save_to_file(self: Self, seq_dir: str):
        if self.the_type == NNRecordType.LIST:
            [each.save_to_file(seq_dir) for each in self.value]
            return
        elif self.the_type == NNRecordType.MAP:
            [each.save_to_file(seq_dir) for each in self.value.values()]
            return
        elif self.the_type == NNRecordType.ARRAY:
            self.save_to_file_pb(seq_dir)
        elif self.the_type == NNRecordType.OP_ITEM:
            self.value.save_to_file(seq_dir)

    def save_to_file_pb(self: Self, seq_dir: str):
        serialized = self.serialize_pb()

        filename = f'{self.data_id}.pb'

        out_filename = os.sep.join([cfg.config['output_dir'], seq_dir, filename])
        if os.path.exists(out_filename):
            cfg.logger.warning(f'save_to_file_pb: (record) file already exists: seq_dir: {seq_dir} data_id: {self.data_id} out_filename: {out_filename}')  # noqa

        utils.ensure_dir(out_filename)
        with open(out_filename, 'wb') as f:
            serialized_bytes = serialized.SerializeToString()
            serialized_str = base64.b64encode(serialized_bytes)
            f.write(serialized_str)
