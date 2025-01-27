# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np

from . import constants
from . import nnextractor_pb2


def serialize_ndarray(ary: np.ndarray, dtype: Optional[str] = None) -> nnextractor_pb2.NDArray:
    if dtype is None:
        dtype = _dtype(ary.dtype)

    if ary.ndim == 1:
        return _serialize_ndarray_1d(ary, dtype)

    lists = [serialize_ndarray(ary[idx], dtype) for idx in range(ary.shape[0])]
    dtype = constants.pb_type_to_pb_list_type(dtype)
    return nnextractor_pb2.NDArray(type=dtype, lists=lists)


def _dtype(dtype: np.dtype) -> str:
    the_type = ''

    if np.isdtype(dtype, np.float64):
        the_type = constants.PB_FLOAT64
    elif np.isdtype(dtype, np.int64):
        the_type = constants.PB_INT64
    elif np.isdtype(dtype, np.bool):
        the_type = constants.PB_BOOL
    elif np.isdtype(dtype, np.int32):
        the_type = constants.PB_INT32
    elif np.isdtype(dtype, np.float32):
        the_type = constants.PB_FLOAT32

    return the_type


def _meta_type(dtype: np.dtype) -> str:

    dtype_str = _dtype(dtype)

    return constants.PB_META_TYPE_MAP[dtype_str]


def _serialize_ndarray_1d(ary: np.ndarray, the_type: str) -> nnextractor_pb2.NDArray:
    bools = None
    int32s = None
    int64s = None
    float32s = None
    float64s = None

    if the_type == constants.PB_FLOAT64:
        float64s = ary.tolist()
    elif the_type == constants.PB_INT64:
        int64s = ary.tolist()
    elif the_type == constants.PB_BOOL:
        bools = ary.tolist()
    elif the_type == constants.PB_INT32:
        int32s = ary.tolist()
    elif the_type == constants.PB_FLOAT32:
        float32s = ary.tolist()

    return nnextractor_pb2.NDArray(type=the_type, bools=bools, int32s=int32s, int64s=int64s, float32s=float32s, float64s=float64s, lists=None)


def deserialize_ndarray(ary_pb: nnextractor_pb2.NDArray, is_first: bool = True):
    the_ary = None
    dtype = None

    if ary_pb.type == constants.PB_LIST_FLOAT64:
        the_ary = [deserialize_ndarray(each, is_first=False) for each in ary_pb.lists]
        dtype = np.float64
    elif ary_pb.type == constants.PB_LIST_INT64:
        the_ary = [deserialize_ndarray(each, is_first=False) for each in ary_pb.lists]
        dtype = np.int64
    elif ary_pb.type == constants.PB_BOOL:
        the_ary = ary_pb.bools
        dtype = np.bool
    elif ary_pb.type == constants.PB_INT32:
        the_ary = ary_pb.int32s
        dtype = np.int32
    elif ary_pb.type == constants.PB_INT64:
        the_ary = ary_pb.int64s
        dtype = np.int64
    elif ary_pb.type == constants.PB_FLOAT32:
        the_ary = ary_pb.float32s
        dtype = np.float32
    elif ary_pb.type == constants.PB_FLOAT64:
        the_ary = ary_pb.float64s
        dtype = np.float64
    elif ary_pb.type == constants.PB_LIST_BOOL:
        the_ary = [deserialize_ndarray(each, is_first=False) for each in ary_pb.lists]
        dtype = np.bool
    elif ary_pb.type == constants.PB_LIST_INT32:
        the_ary = [deserialize_ndarray(each, is_first=False) for each in ary_pb.lists]
        dtype = np.int32
    elif ary_pb.type == constants.PB_LIST_FLOAT32:
        the_ary = [deserialize_ndarray(each, is_first=False) for each in ary_pb.lists]
        dtype = np.float32

    if is_first:
        return np.array(the_ary, dtype=dtype)
    else:
        return the_ary


def meta_ndarray(ary: np.ndarray) -> dict:
    return {'shape': ary.shape, 'type': _meta_type(ary.dtype)}
