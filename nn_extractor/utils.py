# -*- coding: utf-8 -*-

import os
from typing import Optional

import copy

from nn_extractor import cfg


def ensure_dir(filename: str):
    the_dirname = os.path.dirname(filename)
    if os.path.exists(the_dirname):
        return
    os.makedirs(the_dirname, exist_ok=True)


def sanitize_slice_sar(slice_sar: list[slice]):
    slice_sar = [sanitize_slice(each) for each in slice_sar]
    return slice_sar


def sanitize_slice(
    the_slice: int | slice | tuple[Optional[int], Optional[int], Optional[int]] | list[Optional[int]]  # noqa
) -> Optional[slice]:
    if isinstance(the_slice, int):
        # retaining the result as int as indication to reduce dimension.
        return the_slice
    if isinstance(the_slice, slice):
        return the_slice

    if len(the_slice) > 3:
        raise Exception(f'sanitize_slice: invalid slice: the_slice: {the_slice}')

    new_slice = copy.deepcopy(the_slice)
    if len(new_slice) == 1:
        new_slice += [None, None]
    elif len(the_slice) == 2:
        new_slice += [None]

    return slice(new_slice[0], new_slice[1], new_slice[2])
