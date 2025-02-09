# -*- coding: utf-8 -*-

from enum import Enum
from typing import Optional, TypedDict, NamedTuple
import numpy as np
import pyarrow as pa
import io

from . import constants
from . import nnextractor_pb2

type NDArray = np.ndarray


class NDArrayType(Enum):
    UNSPECIFIED = nnextractor_pb2.NDArrayType.NDA_UNSPECIFIED

    BOOL = nnextractor_pb2.NDArrayType.NDA_BOOL
    INT8 = nnextractor_pb2.NDArrayType.NDA_INT8
    INT16 = nnextractor_pb2.NDArrayType.NDA_INT16
    INT32 = nnextractor_pb2.NDArrayType.NDA_INT32
    INT64 = nnextractor_pb2.NDArrayType.NDA_INT64
    FLOAT32 = nnextractor_pb2.NDArrayType.NDA_FLOAT32
    FLOAT64 = nnextractor_pb2.NDArrayType.NDA_FLOAT64
    FLOAT16 = nnextractor_pb2.NDArrayType.NDA_FLOAT16

    UINT8 = nnextractor_pb2.NDArrayType.NDA_UINT8
    UINT16 = nnextractor_pb2.NDArrayType.NDA_UINT16
    UINT32 = nnextractor_pb2.NDArrayType.NDA_UINT32
    UINT64 = nnextractor_pb2.NDArrayType.NDA_UINT64

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return _ND_ARRAY_TYPE_STR_MAP[self]


_ND_ARRAY_TYPE_STR_MAP = {
    NDArrayType.UNSPECIFIED: constants.META_UNSPECIFIED,

    NDArrayType.BOOL: constants.META_BOOL,
    NDArrayType.INT8: constants.META_INT8,
    NDArrayType.INT16: constants.META_INT16,
    NDArrayType.INT32: constants.META_INT32,
    NDArrayType.INT64: constants.META_INT64,
    NDArrayType.FLOAT32: constants.META_FLOAT32,
    NDArrayType.FLOAT64: constants.META_FLOAT64,
    NDArrayType.FLOAT16: constants.META_FLOAT16,

    NDArrayType.UINT8: constants.META_UINT8,
    NDArrayType.UINT16: constants.META_UINT16,
    NDArrayType.UINT32: constants.META_UINT32,
    NDArrayType.UINT64: constants.META_UINT64,
}


class ArrayNDArray(NamedTuple):
    shape: list[int]
    ary: pa.Array


class MetaNDArray(TypedDict):
    shape: list[int]
    the_type: str


def serialize_ndarray_pk(ndarray: np.ndarray, dtype: Optional[NDArrayType] = None) -> np.ndarray:
    return ndarray


def deserialize_ndarray_pk(ndarray_pk: np.ndarray, is_first: bool = True) -> np.ndarray | list:
    return ndarray_pk


def serialize_ndarray_pb(ndarray: np.ndarray, dtype: Optional[NDArrayType] = None) -> nnextractor_pb2.NDArray:
    the_shape = ndarray.shape
    the_table = pa.table([pa.array(ndarray.flatten())], names=[''])
    with io.BytesIO() as f:
        with pa.ipc.new_stream(f, the_table.schema) as writer:
            writer.write_table(the_table)

        the_bytes = f.getvalue()

    return nnextractor_pb2.NDArray(shape=the_shape, the_bytes=the_bytes)


def deserialize_ndarray_pb(ndarray_pb: nnextractor_pb2.NDArray) -> np.ndarray:
    the_shape = ndarray_pb.shape
    the_bytes = ndarray_pb.the_bytes
    with io.BytesIO(the_bytes) as f:
        with pa.ipc.open_stream(f) as reader:
            array_1d: np.ndarray = reader.read_pandas().to_numpy()
    ndarray = array_1d.reshape(*the_shape)
    return ndarray


def meta_ndarray(ary: np.ndarray) -> MetaNDArray:
    return MetaNDArray(shape=ary.shape, the_type=str(_dtype(ary.dtype)))


def _dtype(dtype: np.dtype) -> NDArrayType:
    if np.isdtype(dtype, np.float64):
        return NDArrayType.FLOAT64
    elif np.isdtype(dtype, np.int64):
        return NDArrayType.INT64
    elif np.isdtype(dtype, np.bool):
        return NDArrayType.BOOL
    elif np.isdtype(dtype, np.int8):
        return NDArrayType.INT8
    elif np.isdtype(dtype, np.int16):
        return NDArrayType.INT16
    elif np.isdtype(dtype, np.int32):
        return NDArrayType.INT32
    elif np.isdtype(dtype, np.float32):
        return NDArrayType.FLOAT32
    elif np.isdtype(dtype, np.float16):
        return NDArrayType.FLOAT16
    elif np.isdtype(dtype, np.uint8):
        return NDArrayType.UINT8
    elif np.isdtype(dtype, np.uint16):
        return NDArrayType.UINT16
    elif np.isdtype(dtype, np.uint32):
        return NDArrayType.UINT32
    elif np.isdtype(dtype, np.uint64):
        return NDArrayType.UINT64
    else:
        raise Exception(f'_dtype: type error: {dtype}')
