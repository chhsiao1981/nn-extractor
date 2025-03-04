# -*- coding: utf-8 -*-

import copy

from typing import Optional
from nn_extractor import cfg
from nn_extractor.types import NNTensor


def prepend_list[T](
    the_list: list[T],
    img: NNTensor,
    the_default: T,
) -> list[T]:
    '''
    assuming that img.ndim >= len(cropping)
    '''
    if img.ndim < len(the_list):
        cfg.logger.warning(f'prepend_list: img.ndim < the_list: img.ndim: {img.ndim} the_list: {len(the_list)}')  # noqa
        return the_list

    to_prepend = img.ndim - len(the_list)

    return [the_default] * to_prepend + the_list


def sanitize_slice(
    the_slice: slice | tuple[Optional[int], Optional[int], Optional[int]] | list[Optional[int]]  # noqa
) -> Optional[slice]:
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
