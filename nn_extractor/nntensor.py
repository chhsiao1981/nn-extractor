# -*- coding: utf-8 -*-

from enum import Enum
from typing import Optional, Any
import numpy as np
import pyarrow as pa
import io
import torch

from . import constants
from . import nnextractor_pb2

from .types import MetaNNTensor, NNTensor


class NNTensorType(Enum):
    UNSPECIFIED = nnextractor_pb2.NNTensorType.NNT_UNSPECIFIED

    BOOL = nnextractor_pb2.NNTensorType.NNT_BOOL
    INT8 = nnextractor_pb2.NNTensorType.NNT_INT8
    INT16 = nnextractor_pb2.NNTensorType.NNT_INT16
    INT32 = nnextractor_pb2.NNTensorType.NNT_INT32
    INT64 = nnextractor_pb2.NNTensorType.NNT_INT64
    FLOAT32 = nnextractor_pb2.NNTensorType.NNT_FLOAT32
    FLOAT64 = nnextractor_pb2.NNTensorType.NNT_FLOAT64
    FLOAT16 = nnextractor_pb2.NNTensorType.NNT_FLOAT16

    UINT8 = nnextractor_pb2.NNTensorType.NNT_UINT8
    UINT16 = nnextractor_pb2.NNTensorType.NNT_UINT16
    UINT32 = nnextractor_pb2.NNTensorType.NNT_UINT32
    UINT64 = nnextractor_pb2.NNTensorType.NNT_UINT64

    BFLOAT16 = nnextractor_pb2.NNTensorType.NNT_BFLOAT16

    COMPLEX32 = nnextractor_pb2.NNTensorType.NNT_COMPLEX32
    COMPLEX64 = nnextractor_pb2.NNTensorType.NNT_COMPLEX64
    COMPLEX128 = nnextractor_pb2.NNTensorType.NNT_COMPLEX128

    UQINT8 = nnextractor_pb2.NNTensorType.NNT_UQINT8
    QINT8 = nnextractor_pb2.NNTensorType.NNT_QINT8
    QINT32 = nnextractor_pb2.NNTensorType.NNT_QINT32
    UQINT4 = nnextractor_pb2.NNTensorType.NNT_UQINT4

    FLOAT8_E4M3FN = nnextractor_pb2.NNTensorType.NNT_FLOAT8_E4M3FN
    FLOAT8_E5M2 = nnextractor_pb2.NNTensorType.NNT_FLOAT8_E5M2

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return _TENSOR_TYPE_STR_MAP[self]


_TENSOR_TYPE_STR_MAP = {
    NNTensorType.UNSPECIFIED: constants.META_UNSPECIFIED,

    NNTensorType.BOOL: constants.META_BOOL,
    NNTensorType.INT8: constants.META_INT8,
    NNTensorType.INT16: constants.META_INT16,
    NNTensorType.INT32: constants.META_INT32,
    NNTensorType.INT64: constants.META_INT64,
    NNTensorType.FLOAT32: constants.META_FLOAT32,
    NNTensorType.FLOAT64: constants.META_FLOAT64,
    NNTensorType.FLOAT16: constants.META_FLOAT16,

    NNTensorType.UINT8: constants.META_UINT8,
    NNTensorType.UINT16: constants.META_UINT16,
    NNTensorType.UINT32: constants.META_UINT32,
    NNTensorType.UINT64: constants.META_UINT64,
    NNTensorType.BFLOAT16: constants.META_BFLOAT16,

    NNTensorType.COMPLEX32: constants.META_COMPLEX32,
    NNTensorType.COMPLEX64: constants.META_COMPLEX64,
    NNTensorType.COMPLEX128: constants.META_COMPLEX128,

    NNTensorType.UQINT8: constants.META_UQINT8,
    NNTensorType.QINT8: constants.META_QINT8,
    NNTensorType.QINT32: constants.META_QINT32,
    NNTensorType.UQINT4: constants.META_UQINT4,

    NNTensorType.FLOAT8_E4M3FN: constants.META_FLOAT8_E4M3FN,
    NNTensorType.FLOAT8_E5M2: constants.META_FLOAT8_E5M2,
}


def isinstance_nntensor(obj: Any) -> bool:
    return isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor)


def eq_nntensor(nntensor: NNTensor, other: Any) -> bool:
    if not isinstance(other, type(nntensor)):
        return False

    if nntensor.shape != other.shape:
        return False

    is_eq: NNTensor = nntensor == other

    return is_eq.all()


def serialize_nntensor_pb(
    nntensor: NNTensor,
    dtype: Optional[NNTensorType] = None,
) -> nnextractor_pb2.NNTensor:
    if isinstance(nntensor, torch.Tensor):
        nntensor = nntensor.detach().to('cpu').numpy()

    the_shape = nntensor.shape
    the_table = pa.table([pa.array(nntensor.flatten())], names=[''])
    with io.BytesIO() as f:
        with pa.ipc.new_stream(f, the_table.schema) as writer:
            writer.write_table(the_table)

        the_bytes = f.getvalue()

    return nnextractor_pb2.NNTensor(shape=the_shape, the_bytes=the_bytes)


def deserialize_nntensor_pb(nntensor_pb: nnextractor_pb2.NNTensor) -> NNTensor:
    the_shape = nntensor_pb.shape
    the_bytes = nntensor_pb.the_bytes
    with io.BytesIO(the_bytes) as f:
        with pa.ipc.open_stream(f) as reader:
            array_1d: np.ndarray = reader.read_pandas().to_numpy()
    nntensor = array_1d.reshape(*the_shape)
    return nntensor


def meta_nntensor(nntensor: NNTensor) -> MetaNNTensor:
    return MetaNNTensor(shape=nntensor.shape, the_type=str(dtype(nntensor)))


def dtype(nntensor: NNTensor) -> NNTensorType:
    if isinstance(nntensor, np.ndarray):
        return ndarray_dtype(nntensor.dtype)

    if isinstance(nntensor, torch.Tensor):
        return tensor_type(nntensor.dtype)

    raise Exception(f'invalid nntensor type: {type(nntensor)}')


def ndarray_dtype(the_dtype: np.dtype) -> NNTensorType:
    if np.isdtype(the_dtype, np.float64):
        return NNTensorType.FLOAT64
    elif np.isdtype(the_dtype, np.int64):
        return NNTensorType.INT64
    elif np.isdtype(the_dtype, np.bool):
        return NNTensorType.BOOL
    elif np.isdtype(the_dtype, np.int8):
        return NNTensorType.INT8
    elif np.isdtype(the_dtype, np.int16):
        return NNTensorType.INT16
    elif np.isdtype(the_dtype, np.int32):
        return NNTensorType.INT32
    elif np.isdtype(the_dtype, np.float32):
        return NNTensorType.FLOAT32
    elif np.isdtype(the_dtype, np.float16):
        return NNTensorType.FLOAT16
    elif np.isdtype(the_dtype, np.uint8):
        return NNTensorType.UINT8
    elif np.isdtype(the_dtype, np.uint16):
        return NNTensorType.UINT16
    elif np.isdtype(the_dtype, np.uint32):
        return NNTensorType.UINT32
    elif np.isdtype(the_dtype, np.uint64):
        return NNTensorType.UINT64
    else:
        raise Exception(f'_dtype: type error: {the_dtype}')


def tensor_type(the_tensor_type: torch.dtype) -> NNTensorType:
    if the_tensor_type == torch.float64:
        return NNTensorType.FLOAT64
    elif the_tensor_type == torch.int64:
        return NNTensorType.INT64
    elif the_tensor_type == torch.bool:
        return NNTensorType.BOOL
    elif the_tensor_type == torch.int8:
        return NNTensorType.INT8
    elif the_tensor_type == torch.int16:
        return NNTensorType.INT16
    elif the_tensor_type == torch.int32:
        return NNTensorType.INT32
    elif the_tensor_type == torch.float32:
        return NNTensorType.FLOAT32
    elif the_tensor_type == torch.float16:
        return NNTensorType.FLOAT16
    elif the_tensor_type == torch.bfloat16:
        return NNTensorType.BFLOAT16
    elif the_tensor_type == torch.uint8:
        return NNTensorType.UINT8
    elif the_tensor_type == torch.uint16:
        return NNTensorType.UINT16
    elif the_tensor_type == torch.uint32:
        return NNTensorType.UINT32
    elif the_tensor_type == torch.uint64:
        return NNTensorType.UINT64
    else:
        raise Exception(f'tensor_type: type error: {the_tensor_type}')
