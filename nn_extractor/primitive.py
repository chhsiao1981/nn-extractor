# -*- coding: utf-8 -*-

from typing import Any

import numpy as np

from . import nnextractor_pb2
from . import nntensor


from .types import Primitive, MetaPrimitive


def eq_primitive(primitive: Primitive, other: Any) -> bool:
    return primitive == other


def serialize_primitive_pb(primitive: Primitive) -> nnextractor_pb2.Primitive:
    if primitive is None:
        return nnextractor_pb2.Primitive(the_type=nnextractor_pb2.PrimitiveType.P_NULL)
    elif isinstance(primitive, float) or isinstance(primitive, np.float64):
        return nnextractor_pb2.Primitive(
            the_type=nnextractor_pb2.PrimitiveType.P_F64,
            f64=primitive)
    elif isinstance(primitive, int) or isinstance(primitive, np.int64):
        return nnextractor_pb2.Primitive(
            the_type=nnextractor_pb2.PrimitiveType.P_I64,
            i64=primitive)
    elif isinstance(primitive, bool) or isinstance(primitive, np.bool):
        return nnextractor_pb2.Primitive(
            the_type=nnextractor_pb2.PrimitiveType.P_BOOL,
            b=primitive)
    elif isinstance(primitive, str):
        return nnextractor_pb2.Primitive(
            the_type=nnextractor_pb2.PrimitiveType.P_STR,
            s=primitive)
    elif isinstance(primitive, np.float32):
        return nnextractor_pb2.Primitive(
            the_type=nnextractor_pb2.PrimitiveType.P_F32,
            f32=primitive)
    elif isinstance(primitive, np.int32):
        return nnextractor_pb2.Primitive(
            the_type=nnextractor_pb2.PrimitiveType.P_I32,
            i32=primitive)
    else:
        raise Exception(f'type error: the_type: {type(primitive)}')


def deserialize_primitive_pb(primitive_pb: nnextractor_pb2.Primitive) -> Primitive:
    the_type = primitive_pb.the_type
    if the_type == nnextractor_pb2.PrimitiveType.P_NULL:
        return None
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
        return nntensor.deserialize_nntensor_pb(primitive_pb.ndarray)
    else:
        raise Exception(f'type error: the_type: {the_type}')


def meta_primitive(primitive: Primitive) -> MetaPrimitive:
    if isinstance(primitive, np.bool):
        return primitive.item()
    if isinstance(primitive, np.float16):
        return primitive.item()
    if isinstance(primitive, np.float64):
        return primitive.item()
    if isinstance(primitive, np.float32):
        return primitive.item()
    if isinstance(primitive, np.int8):
        return primitive.item()
    if isinstance(primitive, np.int16):
        return primitive.item()
    if isinstance(primitive, np.int64):
        return primitive.item()
    if isinstance(primitive, np.int32):
        return primitive.item()
    return primitive
