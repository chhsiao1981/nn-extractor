# -*- coding: utf-8 -*-

import os

from nn_extractor import cfg


def ensure_dir(filename: str):
    the_dirname = os.path.dirname(filename)
    if os.path.exists(the_dirname):
        return
    os.makedirs(the_dirname, exist_ok=True)


def slice_spl_to_sar(the_slice: list[slice], the_shape: tuple[int]):
    if len(the_shape) >= 3:
        the_slice_p = the_slice[-2]
        the_slice_l = the_slice[-1]

        the_slice_a = slice_inverse(the_slice_p, the_shape[-2])
        the_slice_r = slice_inverse(the_slice_l, the_shape[-1])

        new_slice = [each for each in the_slice]
        new_slice[-2] = the_slice_a
        new_slice[-1] = the_slice_r
        cfg.logger.info(f'slice_spl_to_sar: the_slice: {the_slice} the_shape: {the_shape} new_slice: {new_slice}')  # noqa
        return new_slice
    else:
        cfg.logger.info(f'slice_spl_to_sar: the_shape < 3: the_slice: {the_slice} the_shape: {the_shape}')  # noqa

    return the_slice


def slice_inverse(the_slice: slice | int, the_dim: int) -> slice | int:
    if isinstance(the_slice, int):
        return the_dim - 1 - the_slice

    the_stop = None if the_slice.start is None else the_dim - 1 - the_slice.start
    the_start = None if the_slice.stop is None else the_dim - 1 - the_slice.stop

    the_step = the_slice.step

    return slice(the_start, the_stop, the_step)
