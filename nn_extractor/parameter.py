# -*- coding: utf-8 -*-

from typing import NamedTuple, Any

from . import record
from .record import Record

from . import nnextractor_pb2


class Parameter(NamedTuple):
    name: str
    record: Record


def is_same_parameter(params_a: Parameter, params_b: Parameter) -> bool:
    if not _is_valid_type(params_a):
        return False

    if not _is_valid_type(params_b):
        return False

    if len(params_a) != 2:
        return False

    if len(params_b) != 2:
        return False

    if params_a[0] != params_b[0]:
        return False

    if not record.is_same_record(params_a[1], params_b[1]):
        return False

    return True


def _is_valid_type(params: Any) -> bool:
    if isinstance(params, tuple):
        return True

    return False


def serialize_parameter(param: Parameter) -> nnextractor_pb2.Parameter:
    record_pb = record.serialize_record(param[1])
    return nnextractor_pb2.Parameter(name=param[0], record=record_pb)


def deserialize_parameter(params_pb: nnextractor_pb2.Parameter) -> Parameter:
    the_record = record.deserialize_record(params_pb.record)
    return Parameter(name=params_pb.name, record=the_record)


def meta_parameter(params: Parameter) -> dict:
    meta_record = record.meta_record(params[1])

    return {'name': params[0], 'record': meta_record}
