# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, TypedDict, NamedTuple

from .ndarray import NDArray
from .nnrecord import NNRecord, MetaNNRecord, PickleNNRecord

from . import nnextractor_pb2


class MetaNNParameter(TypedDict):
    name: str
    record: MetaNNRecord


class PickleNNParameter(NamedTuple):
    name: str
    record: PickleNNRecord


@dataclass
class NNParameter(object):
    name: str
    record: NNRecord

    def __init__(self, name: str, record: NNRecord | NDArray):
        self.name = name
        if isinstance(record, NNRecord):
            self.record = record
        else:
            self.record = NNRecord(record)

    def __eq__(self: Self, other: Self) -> bool:
        if not isinstance(other, NNParameter):
            return False

        if self.name != other.name:
            return False

        if self.record != other.record:
            return False

        return True

    def serialize_pk(self: Self) -> PickleNNParameter:
        record_pk = self.record.serialize_pk()
        return PickleNNParameter(name=self.name, record=record_pk)

    @classmethod
    def deserialize_pk(cls: Self, params_pk: PickleNNParameter) -> Self:
        the_record = NNRecord.deserialize_pk(params_pk.record)
        return NNParameter(name=params_pk.name, record=the_record)

    def serialize_pb(self: Self) -> nnextractor_pb2.NNParameter:
        record_pb = self.record.serialize_pb()
        return nnextractor_pb2.NNParameter(name=self.name, record=record_pb)

    @classmethod
    def deserialize_pb(cls: Self, params_pb: nnextractor_pb2.NNParameter) -> Self:
        the_record = NNRecord.deserialize_pb(params_pb.record)
        return NNParameter(name=params_pb.name, record=the_record)

    def meta(self: Self) -> MetaNNParameter:
        meta_record = self.record.meta()

        return MetaNNParameter(name=self.name, record=meta_record)
