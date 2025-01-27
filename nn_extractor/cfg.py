# -*- coding: utf-8 -*-

from typing import Callable, Optional, Any
from typing_extensions import NotRequired

import logging

from . import _cfg
from ._cfg import Config as BaseConfig


class Config(BaseConfig):
    name: NotRequired[str]
    output_dir: NotRequired[str]

    is_disable: NotRequired[bool]
    is_debug_config: NotRequired[bool]

    is_nnnode_record_inputs: NotRequired[bool]


config: Config = {
    'name': '',
    'output_dir': './',

    'is_disable': False,
    'is_debug_config': False,

    'is_nnnode_record_inputs': False,

    'is_profile': False,
}

# logger default to root logger
logger: logging.Logger = logging.getLogger()

_NAME = 'nn-extractor'


def init(filename: str = '', extra_params: Optional[dict] = None):
    global config
    global logger

    logger, _config = _cfg.init(_NAME, filename, extra_params=extra_params)
    config.update(_config)

    if config['is_debug_config']:
        logger.debug(f'config: {config}')


def check_disable(func: Callable, default: Any = None):
    def _check_disable(*args, **kwargs):
        if config['is_disable']:
            return default

        return func(*args, **kwargs)

    return _check_disable
