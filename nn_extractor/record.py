# -*- coding: utf-8 -*-

import numpy as np

from . import constants
from . import nnextractor_pb2
from . import ndarray

type Record = tuple[Record] | list[Record] | dict[str, Record] | np.ndarray


def serialize_record(record: Record) -> nnextractor_pb2.Record:
    the_type = constants.PB_ARRAY
    records = None
    the_map = None
    ary = None

    if isinstance(record, np.ndarray):
        the_type = constants.PB_ARRAY
        ary = ndarray.serialize_ndarray(record)
    elif isinstance(record, tuple):
        the_type = constants.PB_TUPLE
        records = [serialize_record(each) for each in record]
    elif isinstance(record, list):
        the_type = constants.PB_LIST
        records = [serialize_record(each) for each in record]
    elif isinstance(record, dict):
        the_type = constants.PB_DICT
        the_map = {k: serialize_record(each) for k, each in record.items()}

    return nnextractor_pb2.Record(type=the_type, array=ary, records=records, the_map=the_map)


def deserialize_record(record_pb: nnextractor_pb2.Record) -> Record:
    the_type = record_pb.type

    if the_type == constants.PB_ARRAY:
        return ndarray.deserialize_ndarray(record_pb.array)
    elif the_type == constants.PB_TUPLE:
        return tuple([deserialize_record(each) for each in record_pb.records])
    elif the_type == constants.PB_LIST:
        return [deserialize_record(each) for each in record_pb.records]
    elif the_type == constants.PB_DICT:
        return {k: deserialize_record(each) for k, each in record_pb.the_map.items()}


def meta_record(record: Record):
    if isinstance(record, np.ndarray):
        return ndarray.meta_ndarray(record)
    elif isinstance(record, tuple):
        return tuple([meta_record(each) for each in record])
    elif isinstance(record, list):
        return [meta_record(each) for each in record]
    elif isinstance(record, dict):
        return {k: meta_record(each) for k, each in record.items()}


def is_same_record(record_a: Record, record_b: Record) -> bool:
    # type
    if not _is_valid_type(record_a):
        return False

    if not _is_valid_type(record_b):
        return False

    if type(record_a) != type(record_b):
        return False

    # ndarray
    if isinstance(record_a, np.ndarray):
        return (record_a == record_b).all() and np.isdtype(record_a.dtype, record_a.dtype)

    # tuple
    if isinstance(record_a, tuple):
        if len(record_a) != len(record_b):
            return False

        for idx, each in enumerate(record_a):
            if not is_same_record(each, record_b[idx]):
                return False
        return True

    # list
    if isinstance(record_a, list):
        if len(record_a) != len(record_b):
            return False

        for idx, each in enumerate(record_a):
            if not is_same_record(each, record_b[idx]):
                return False
        return True

    # dict
    if isinstance(record_a, dict):
        for idx, each in record_a.items():
            if idx not in record_b:
                return False
            if not is_same_record(each, record_b[idx]):
                return False

        for key in record_b.keys():
            if key not in record_a:
                return False
        return True

    return False


def _is_valid_type(rec: Record) -> bool:
    if isinstance(rec, np.ndarray):
        return True
    if isinstance(rec, list):
        return True
    if isinstance(rec, tuple):
        return True
    if isinstance(rec, dict):
        return True

    return False
