# -*- coding: utf-8 -*-


def pb_type_to_pb_list_type(pb_type: str) -> str:
    return pb_type.upper()


def pb_list_type_to_pb_type(pb_list_type: str) -> str:
    return pb_list_type.lower()


ACTIVATIONS = 'activations'
GRADIENTS = 'gradients'


PB_BOOL = 'b'
PB_INT32 = 'h'
PB_INT64 = 'i'
PB_FLOAT32 = 'e'
PB_FLOAT64 = 'f'

PB_LIST_BOOL = pb_type_to_pb_list_type(PB_BOOL)
PB_LIST_INT32 = pb_type_to_pb_list_type(PB_INT32)
PB_LIST_INT64 = pb_type_to_pb_list_type(PB_INT64)
PB_LIST_FLOAT32 = pb_type_to_pb_list_type(PB_FLOAT32)
PB_LIST_FLOAT64 = pb_type_to_pb_list_type(PB_FLOAT64)

PB_ARRAY = 'a'
PB_TUPLE = 't'
PB_LIST = 'l'
PB_DICT = 'd'

METADATA_FILENAME = '.metadata'

META_BOOL = 'bool'
META_INT32 = 'i32'
META_INT64 = 'i64'
META_FLOAT32 = 'f32'
META_FLOAT64 = 'f64'

META_PB_MAP = {
    META_BOOL: PB_BOOL,
    META_INT32: PB_INT32,
    META_INT64: PB_INT64,
    META_FLOAT32: PB_FLOAT32,
    META_FLOAT64: PB_FLOAT64,
}

PB_META_TYPE_MAP = {
    PB_BOOL: META_BOOL,
    PB_LIST_BOOL: META_BOOL,

    PB_INT32: META_INT32,
    PB_LIST_INT32: META_INT32,

    PB_INT64: META_INT64,
    PB_LIST_INT64: META_INT64,

    PB_FLOAT32: META_FLOAT32,
    PB_LIST_FLOAT32: META_FLOAT32,

    PB_FLOAT64: META_FLOAT64,
    PB_LIST_FLOAT64: META_FLOAT64,
}
