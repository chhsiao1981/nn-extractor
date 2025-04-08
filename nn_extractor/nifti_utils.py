# -*- coding: utf-8 -*-

import struct


_BYTE_LENGTH_MAP = {
    2: 1,
    4: 2,
    8: 4,
    16: 4,
    32: 8,
    64: 8,
    128: 3,
    256: 1,
    512: 2,
    768: 4,
    1024: 8,
    1280: 8,
    1536: 16,
    1792: 16,
    2048: 32,
    2304: 4,
}

_PACK_MAP = {
    2: '<B',
    4: '<h',
    8: '<i',
    16: '<f',
    32: '',
    64: '<d',
    128: '<BBB',
    256: '<b',
    512: '<H',
    768: '<I',
    1024: '<q',
    1280: '<Q',
    1536: '',
    1792: '<dd',
    2048: '',
    2304: '<BBBB',
}


def get_bytes(filename: str):
    with open(filename, 'rb') as f:
        return f.read()


def get_value(data_bytes: bytes, r: int, a: int, s: int, dims_ras: list[int], data_type: int):
    '''
    NIFTI is column-major

    https://afni.nimh.nih.gov/pub/dist/doc/nifti/nifti1_rationale.html#:~:text=In%20terms%20of%20the%20actual%20image%20data%20storage.
    Data Dimensionality

    In terms of the actual image data storage, this makes no difference, since the data at the 5D index (a,b,c,d,e) is to be stored at byte offset
    (a+b*dim[1]+c*dim[1]*dim[2]+d*dim[1]*dim[2]*dim[3]+e*dim[1]*dim[2]*dim[3]*dim[4])*bitpix/8

    '''
    byte_length = _BYTE_LENGTH_MAP[data_type]
    pack_str = _PACK_MAP[data_type]
    the_idx = (r + a * dims_ras[0] + s * dims_ras[0] * dims_ras[1]) * byte_length
    return struct.unpack(pack_str, data_bytes[the_idx:(the_idx + byte_length)])
