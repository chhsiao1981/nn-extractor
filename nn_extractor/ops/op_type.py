# -*- coding: utf-8 -*-

from enum import Enum
from .. import nnextractor_pb2


class OpType(Enum):
    '''
    https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0
    '''

    UNSPECIFIED = nnextractor_pb2.OpType.O_UNSPECIFIED

    UNKNOWN = nnextractor_pb2.OpType.O_UNKNOWN
    OTHER = nnextractor_pb2.OpType.O_OTHER
    NONE = nnextractor_pb2.OpType.O_NONE

    CROP = nnextractor_pb2.OpType.O_CROP

    PAD = nnextractor_pb2.OpType.O_PAD

    FLIP = nnextractor_pb2.OpType.O_FLIP

    ORIGIN = nnextractor_pb2.OpType.O_ORIGIN

    SPACING = nnextractor_pb2.OpType.O_SPACING

    DIRECTION = nnextractor_pb2.OpType.O_DIRECTION

    AFFINE = nnextractor_pb2.OpType.O_AFFINE

    GEO_IDENTITY = nnextractor_pb2.OpType.O_GEO_IDENTITY

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return self.name.lower()


OP_TYPE_FURTHER_INTEGRATION_SET = set([
    OpType.CROP,
    OpType.PAD,
    OpType.FLIP,
    OpType.ORIGIN,
    OpType.SPACING,
    OpType.DIRECTION,
    OpType.AFFINE,
])
