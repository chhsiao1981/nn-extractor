# -*- coding: utf-8 -*-

from . import cfg

import time

from .types import ProfileValue

_PROFILE_POOL: dict[str, ProfileValue] = {}


def profile_start(name):
    if name not in _PROFILE_POOL:
        _PROFILE_POOL[name] = {'start': 0, 'diff': 0, 'count': 0}

    _PROFILE_POOL[name]['start'] = time.time()


def profile_stop(name, is_step=True):
    the_now = time.time()
    if name not in _PROFILE_POOL:
        return

    last_diff = the_now - _PROFILE_POOL[name]['start']
    _PROFILE_POOL[name]['last_diff'] = last_diff
    _PROFILE_POOL[name]['diff'] += last_diff

    if is_step:
        _PROFILE_POOL[name]['count'] += 1


def reset():
    keys = list(_PROFILE_POOL.keys())
    for k in keys:
        del _PROFILE_POOL[k]


def report(prompt='', is_last_diff_only=False):
    keys = list(_PROFILE_POOL.keys())

    if is_last_diff_only:
        for idx, k in enumerate(keys):
            last_diff = _PROFILE_POOL[k]['last_diff']
            count = _PROFILE_POOL[k]['count']
            cfg.logger.info(f'[PROFILE/{prompt}] ({idx} / {len(keys)}) {k}: last-diff: {last_diff} count: {count}')  # noqa
        return

    for idx, k in enumerate(keys):
        diff = _PROFILE_POOL[k]['diff']
        count = _PROFILE_POOL[k]['count']
        avg = float(diff) / float(count)
        cfg.logger.info(
            f'[PROFILE/{prompt}] ({idx} / {len(keys)}) {k}: total: {diff} count: {count} avg: {avg:.03f}')  # noqa
