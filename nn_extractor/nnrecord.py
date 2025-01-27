# -*- coding: utf-8 -*-

from enum import Enum
from dataclasses import dataclass
from typing import Self, Any, NamedTuple, Optional, TypedDict

import numpy as np
from . import nnextractor_pb2
from . import ndarray
from .ndarray import MetaNDArray, NDArray


class NNRecordType(Enum):
    UNSPECIFIED = nnextractor_pb2.NNRecordType.NNR_UNSPECIFIED
    ARRAY = nnextractor_pb2.NNRecordType.NNR_ARRAY
    LIST = nnextractor_pb2.NNRecordType.NNR_LIST
    DICT = nnextractor_pb2.NNRecordType.NNR_DICT

    def __int__(self) -> int:
        return self.value


type MetaNonameNNRecord = MetaNDArray | list[MetaNonameNNRecord] | dict[str, MetaNonameNNRecord]


class MetaNamedNNRecord(TypedDict):
    name: str
    record: MetaNonameNNRecord


type MetaNNRecord = MetaNonameNNRecord | MetaNamedNNRecord


class PickleNNRecord(NamedTuple):
    name: str
    the_type: NNRecordType
    ndarray: Optional[np.ndarray] = None
    records: Optional[list[Self]] = None
    the_map: Optional[dict[str, Self]] = None


@dataclass
class NNRecord(object):
    name: str
    the_type: NNRecordType = NNRecordType.UNSPECIFIED
    ndarray: Optional[np.ndarray] = None
    records: Optional[list[Self]] = None
    the_map: Optional[dict[str, Self]] = None

    def __init__(
        self: Self,

        value: Optional[NDArray | list[Self] | dict[str, Self]] = None,

        name: str = '',

        the_type: Optional[NNRecordType] = None,
        ndarray: Optional[np.ndarray] = None,
        records: Optional[list[Self]] = None,
        the_map: Optional[dict[str, Self]] = None
    ):
        '''
        NNRecord

        It is usual that we have only the value and the name is already determined by nnnode,
        making the name as extremely unnecessary variable.
        We put value as the 1st parameter.
        '''

        self.name = name

        if value is None and the_type is not None:
            self._init_with_deserialization(the_type, ndarray, records, the_map)
            return

        if isinstance(value, np.ndarray):
            self.the_type = NNRecordType.ARRAY
            self.ndarray = value
        elif isinstance(value, list) or isinstance(value, tuple):
            self.the_type = NNRecordType.LIST
            self.records = [NNRecord._single_nnrecord(each) for each in value]
        elif isinstance(value, dict):
            self.the_type = NNRecordType.DICT
            self.the_map = {k: NNRecord._single_nnrecord(each) for k, each in value.items()}
        else:
            raise Exception(f'NNRecord.__init__: invalid type: {type(value)}')

    def _init_with_deserialization(
        self: Self,

        the_type: NNRecordType,
        ndarray: Optional[np.ndarray] = None,
        records: Optional[list[Self]] = None,
        the_map: Optional[dict[str, Self]] = None
    ):
        self.the_type = the_type
        self.ndarray = ndarray
        self.records = records
        self.the_map = the_map

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, NNRecord):
            return False
        if self.name != other.name:
            return False
        if self.the_type != other.the_type:
            return False
        if self.the_type == NNRecordType.ARRAY:
            return (self.ndarray == other.ndarray).all()
        if self.the_type == NNRecordType.LIST:
            if len(self.records) != len(other.records):
                return False
            for idx, each in enumerate(self.records):
                if each != other.records[idx]:
                    return False
            return True
        if self.the_type == NNRecordType.DICT:
            if len(self.the_map) != len(other.the_map):
                return False
            for key, each in self.the_map.items():
                if each != other.the_map[key]:
                    return False
            return True

        return False

    def serialize_pk(self: Self) -> PickleNNRecord:
        records = None if self.records is None \
            else [each.serialize_pk() for each in self.records]
        the_map = None if self.the_map is None \
            else {k: each.serialize_pk() for k, each in self.the_map.items()}
        return PickleNNRecord(
            name=self.name,
            the_type=self.the_type,
            ndarray=self.ndarray,
            records=records,
            the_map=the_map,
        )

    @classmethod
    def deserialize_pk(cls: Self, record_pk: PickleNNRecord) -> Self:
        records = None if record_pk.records is None \
            else [cls.deserialize_pk(each) for each in record_pk.records]
        the_dict = None if record_pk.the_map is None \
            else {k: cls.deserialize_pb(each) for k, each in record_pk.the_map.items()}
        return NNRecord(
            name=record_pk.name,
            the_type=record_pk.the_type,
            ndarray=record_pk.ndarray,
            records=records,
            the_map=the_dict,
        )

    def serialize_pb(self: Self) -> nnextractor_pb2.NNRecord:
        if self.the_type == NNRecordType.ARRAY:
            serialized_ndarray = ndarray.serialize_ndarray_pb(self.ndarray)
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), ndarray=serialized_ndarray)
        elif self.the_type == NNRecordType.LIST:
            serialized_list = [each.serialize_pb() for each in self.records]
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), records=serialized_list)
        elif self.the_type == NNRecordType.DICT:
            serialized_dict = {k: each.serialize_pb() for k, each in self.the_map.items()}
            return nnextractor_pb2.NNRecord(the_type=int(self.the_type), the_map=serialized_dict)
        else:
            raise Exception(f'NNRecord.serialize: invalid type: {self.the_type}')

    @classmethod
    def deserialize_pb(cls: Self, record_pb: nnextractor_pb2.NNRecord) -> Self:
        if record_pb.the_type == nnextractor_pb2.NNRecordType.NNR_ARRAY:
            the_array = ndarray.deserialize_ndarray_pb(record_pb.ndarray)
            return NNRecord(name=record_pb.name, value=the_array)
        elif record_pb.the_type == nnextractor_pb2.NNRecordType.NNR_LIST:
            the_list = [cls.deserialize_pb(each) for each in record_pb.records]
            return NNRecord(name=record_pb.name, value=the_list)
        elif record_pb.the_type == nnextractor_pb2.NNRecordType.NNR_DICT:
            the_dict = {k: cls.deserialize_pb(each) for k, each in record_pb.the_map.items()}
            return NNRecord(name=record_pb.name, value=the_dict)
        else:
            raise Exception(f'NNRecord.deserialize: invalid type: {record_pb.the_type}')

    def meta(self: Self) -> MetaNNRecord:
        ret: MetaNNRecord = None
        if self.the_type == NNRecordType.ARRAY:
            ret = ndarray.meta_ndarray(self.ndarray)
        elif self.the_type == NNRecordType.LIST:
            ret = [each.meta() for each in self.records]
        elif self.the_type == NNRecordType.DICT:
            ret = {k: each.meta() for k, each in self.the_map.items()}
        else:
            raise Exception(f'NNRecord.meta: invalid type: {self.the_type}')

        if not self.name:
            return ret

        return MetaNamedNNRecord(name=self.name, record=ret)

    @classmethod
    def _single_nnrecord(cls: Self, value: Any) -> Self:
        if isinstance(value, NNRecord):
            return value
        return NNRecord(value)
