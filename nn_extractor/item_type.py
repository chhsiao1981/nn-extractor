# -*- coding: utf-8 -*-

from enum import Enum
from . import nnextractor_pb2


class ItemType(Enum):
    UNSPECIFIED = nnextractor_pb2.ItemType.I_UNSPECIFIED

    '''
    I don't know the type,
    or the types are complicated and I don't want to specify the type.
    I'll just say it is a raw type.
    Let the renderer does whatever it wants.
    (Usually the renderer just presents the data as ndarray)
    '''
    RAW = nnextractor_pb2.ItemType.I_RAW

    '''
    I know the type,
    but it is not listed in the following settings.
    '''
    OTHER = nnextractor_pb2.ItemType.I_OTHER

    '''
    nntensor
    '''
    NNTENSOR = nnextractor_pb2.ItemType.I_NNTENSOR

    '''
    NII type (with origin, spacing, direction, affine)
    '''
    NII = nnextractor_pb2.ItemType.I_NII

    '''
    2D-image, as ndarray
    '''
    IMAGE = nnextractor_pb2.ItemType.I_IMAGE

    '''
    audio (can be presented as sound, or time-domain presentation), as ndarray
    '''
    AUDIO = nnextractor_pb2.ItemType.I_AUDIO

    '''
    spectrogram, as ndarray
    '''
    SPECTROGRAM = nnextractor_pb2.ItemType.I_SPECTROGRAM

    '''
    text
    '''
    TEXT = nnextractor_pb2.ItemType.I_TEXT

    '''
    number
    '''
    NUMBER = nnextractor_pb2.ItemType.I_NUMBER

    '''
    text or number
    '''
    TEXT_NUMBER = nnextractor_pb2.ItemType.I_TEXT_NUMBER

    '''
    null
    '''
    NULL = nnextractor_pb2.ItemType.I_NULL

    '''
    list of items
    '''
    LIST = nnextractor_pb2.ItemType.I_LIST

    '''
    map of items
    '''
    MAP = nnextractor_pb2.ItemType.I_MAP

    '''
    op-item
    '''
    OP_ITEM = nnextractor_pb2.ItemType.I_OP_ITEM

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return self.name.lower()


_ITEM_NDARRAY_TYPE_SET = set([
    ItemType.NNTENSOR,
    ItemType.IMAGE,
    ItemType.AUDIO,
    ItemType.SPECTROGRAM,
])

_ITEM_SKIP_SAVE_FILE_TYPE_SET = set([
    ItemType.UNSPECIFIED,
    ItemType.RAW,
    ItemType.OTHER,
    ItemType.TEXT,
    ItemType.NUMBER,
    ItemType.TEXT_NUMBER,
    ItemType.NULL,

    ItemType.LIST,
    ItemType.MAP,
])


def is_ndarray_type(the_type: ItemType) -> bool:
    return the_type in _ITEM_NDARRAY_TYPE_SET


def is_skip_save_file_type(the_type: ItemType) -> bool:
    return the_type in _ITEM_SKIP_SAVE_FILE_TYPE_SET
