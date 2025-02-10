# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, Optional

from .nnrecord import NNRecord

from . import nnextractor_pb2

from .types import MetaNNParameter, NNTensor


@dataclass
class NNParameter(object):
    name: str
    parameter: NNRecord

    def __init__(
        self,
        name: str,
        parameter: NNRecord | NNTensor,
        data_id: Optional[str] = None,
    ):
        self.name = name
        if isinstance(parameter, NNRecord):
            self.parameter = parameter
        else:
            self.parameter = NNRecord(value=parameter, data_id=data_id)

    def __eq__(self: Self, other: Self) -> bool:
        if not isinstance(other, NNParameter):
            return False

        if self.name != other.name:
            return False

        if self.parameter != other.parameter:
            return False

        return True

    def serialize_pb(self: Self) -> nnextractor_pb2.NNParameter:
        parameter_pb = self.parameter.serialize_pb()
        return nnextractor_pb2.NNParameter(name=self.name, parameter=parameter_pb)

    @classmethod
    def deserialize_pb(cls: Self, parameter_pb: nnextractor_pb2.NNParameter) -> Self:
        the_parameter = NNRecord.deserialize_pb(parameter_pb.parameter)
        return NNParameter(name=parameter_pb.name, parameter=the_parameter)

    def meta(self: Self) -> MetaNNParameter:
        meta_parameter = self.parameter.meta()

        return MetaNNParameter(name=self.name, parameter=meta_parameter)

    def save_to_file(self: Self, seq_dir: str):
        self.parameter.save_to_file(seq_dir)
