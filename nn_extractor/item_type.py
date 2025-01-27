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
    ndarray
    Usually present the ndarray as:
    * 1D: numbers
    * 2D: image
    * 3D: 3D - nii
    * > 3D: have users select 3 dimenions and present as 3D nii,
            and show the corresponding info in other dims.
    '''
    NDARRAY = nnextractor_pb2.ItemType.I_NDARRAY

    '''
    3D-nii
    '''
    NII = nnextractor_pb2.ItemType.I_NII

    '''
    2D-image
    '''
    IMAGE = nnextractor_pb2.ItemType.I_IMAGE

    '''
    audio (can be presented as sound, or time-domain presentation)
    '''
    AUDIO = nnextractor_pb2.ItemType.I_AUDIO

    '''
    spectrogram
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

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return self.name.lower()
