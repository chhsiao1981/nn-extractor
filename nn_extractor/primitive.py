# -*- coding: utf-8 -*-

from typing import TypedDict, Any

import numpy as np

from . import nnextractor_pb2
from . import ndarray


type Primitive = None | bool | str | float | int | np.bool | np.float16 | np.float64 | np.float32 | np.int8 | np.int16 | np.int64 | np.int32 | np.ndarray | list[Primitive] | tuple[Primitive] | dict[str, Primitive]  # noqa


class MetaPrimitive(TypedDict):
    the_len: int
    the_type: str


def eq_primitive(primitive: Primitive, other: Any) -> bool:
    if isinstance(primitive, tuple):
        primitive = list(primitive)
    if isinstance(other, tuple):
        other = list(other)

    if _is_np_primitive(primitive):
        primitive = primitive.item()
    if _is_np_primitive(other):
        # other: np.bool | np.float16 | np.float64 | np.float32 | np.int8 | np.int16 | np.int64 | np.int32  # noqa # XXX for type hint
        other = other.item()

    return primitive == other


def _is_np_primitive(item: Any) -> bool:
    return isinstance(item, np.float64) or \
        isinstance(item, np.int64) or \
        isinstance(item, np.float16) or \
        isinstance(item, np.bool) or \
        isinstance(item, np.float32) or \
        isinstance(item, np.int32) or \
        isinstance(item, np.int16) or \
        isinstance(item, np.int8)


def serialize_primitive_pk(primitive: Primitive) -> Primitive:
    return primitive


def deserialize_primitive_pk(primitive_pk: Primitive) -> Primitive:
    return primitive_pk


def serialize_primitive_pb(primitive: Primitive) -> nnextractor_pb2.Primitive:
    if primitive is None:
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_NULL)
    elif isinstance(primitive, list) or isinstance(primitive, tuple):
        serialized_list = [serialize_primitive_pb(each) for each in primitive]
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_LIST, lists=serialized_list)
    elif isinstance(primitive, dict):
        serialized_dict = {k: serialize_primitive_pb(each)for k, each in primitive.items()}
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_DICT, the_map=serialized_dict)
    elif isinstance(primitive, float) or isinstance(primitive, np.float64):
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_F64, f64=primitive)
    elif isinstance(primitive, int) or isinstance(primitive, np.int64):
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_I64, i64=primitive)
    elif isinstance(primitive, bool) or isinstance(primitive, np.bool):
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_BOOL, b=primitive)
    elif isinstance(primitive, str):
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_STR, s=primitive)
    elif isinstance(primitive, np.float32):
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_F32, f32=primitive)
    elif isinstance(primitive, np.int32):
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_I32, i32=primitive)
    elif isinstance(primitive, np.ndarray):
        serialized_ndarray = ndarray.serialize_ndarray_pb(primitive)
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_NDARRAY, ndarray=serialized_ndarray)
    else:
        raise Exception(f'type error: the_type: {type(primitive)}')


def deserialize_primitive_pb(primitive_pb: nnextractor_pb2.Primitive) -> Primitive:
    the_type = primitive_pb.the_type
    if the_type == nnextractor_pb2.PrimitiveType.P_NULL:
        return None
    elif the_type == nnextractor_pb2.PrimitiveType.P_LIST:
        return [deserialize_primitive_pb(each) for each in primitive_pb.lists]
    elif the_type == nnextractor_pb2.PrimitiveType.P_DICT:
        return {k: deserialize_primitive_pb(each) for k, each in primitive_pb.the_map.items()}
    elif the_type == nnextractor_pb2.PrimitiveType.P_BOOL:
        return primitive_pb.b
    elif the_type == nnextractor_pb2.PrimitiveType.P_STR:
        return primitive_pb.s
    elif the_type == nnextractor_pb2.PrimitiveType.P_F64:
        return primitive_pb.f64
    elif the_type == nnextractor_pb2.PrimitiveType.P_F32:
        return np.float32(primitive_pb.f32)
    elif the_type == nnextractor_pb2.PrimitiveType.P_I64:
        return primitive_pb.i64
    elif the_type == nnextractor_pb2.PrimitiveType.P_I32:
        return np.int32(primitive_pb.i32)
    elif the_type == nnextractor_pb2.PrimitiveType.P_NDARRAY:
        return ndarray.deserialize_ndarray_pb(primitive_pb.ndarray)
    else:
        raise Exception(f'type error: the_type: {the_type}')


def meta_primitive(primitive: Primitive) -> Any:
    if isinstance(primitive, np.ndarray):
        return ndarray.meta_ndarray(primitive)
    if isinstance(primitive, list) or isinstance(primitive, tuple):
        return [meta_primitive(val) for val in primitive]
    elif isinstance(primitive, dict):
        return {k: meta_primitive(val) for k, val in primitive.items()}
    else:
        return primitive
