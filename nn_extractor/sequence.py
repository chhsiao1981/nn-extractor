# -*- coding: utf-8 -*-

from typing import Self, TypedDict
from enum import Enum
from dataclasses import dataclass

from . import nnextractor_pb2


class MetaSequence(TypedDict):
    the_type: str
    index: int
    name: str


class SequenceType(Enum):
    UNSPECIFIED = nnextractor_pb2.SequenceType.S_UNSPECIFIED

    INPUTS = nnextractor_pb2.SequenceType.S_INPUTS

    PREPROCESS = nnextractor_pb2.SequenceType.S_PREPROCESS

    FORWARD = nnextractor_pb2.SequenceType.S_FORWARD

    BACKWARD = nnextractor_pb2.SequenceType.S_BACKWARD

    POSTPROCESSES = nnextractor_pb2.SequenceType.S_POSTPROCESSES

    OUTPUTS = nnextractor_pb2.SequenceType.S_OUTPUTS

    EXTRACTORS = nnextractor_pb2.SequenceType.S_EXTRACTORS

    def __int__(self: Self):
        return self.value

    def __str__(self: Self):
        return self.name.lower()


@dataclass
class Sequence(object):
    the_type: SequenceType
    index: int
    name: str = ''

    def serialize(self: Self) -> nnextractor_pb2.Sequence:
        if self.name:
            return nnextractor_pb2.Sequence(the_type=int(self.the_type), index=self.index, name=self.name)
        else:
            return nnextractor_pb2.Sequence(the_type=int(self.the_type), index=self.index)

    @classmethod
    def deserialize(cls: Self, sequence_pb2: nnextractor_pb2.Sequence) -> Self:
        return Sequence(the_type=SequenceType(sequence_pb2.the_type), index=sequence_pb2.index)

    def meta(self: Self) -> MetaSequence:
        return MetaSequence(the_type=str(self.the_type), index=self.index)
