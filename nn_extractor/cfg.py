# -*- coding: utf-8 -*-

from typing import Callable, Optional, Any, TypedDict
from typing_extensions import NotRequired

import importlib
import importlib.metadata

import logging

from . import _cfg

from .server.types import Meta, MetaSummary, MetaNNExtractorSummary, Model, ModelSummary

from .types import MetaNNExtractor


class ServerConfig(TypedDict):
    root_dir: NotRequired[str]
    origins: list[str]


class Config(TypedDict):
    name: NotRequired[str]
    output_dir: NotRequired[str]

    is_disable: NotRequired[bool]
    is_debug_config: NotRequired[bool]

    is_nnnode_record_inputs: NotRequired[bool]

    version: NotRequired[str]

    # server
    server: ServerConfig


config: Config = {
    'name': '',
    'output_dir': './',

    'is_disable': False,
    'is_debug_config': False,

    'is_nnnode_record_inputs': False,

    'is_profile': False,


    'server': {
        'root_dir': '',
        'origins': ['*'],
    }
}


class Global(TypedDict):
    MODEL_LIST: list[Model]
    MODEL_MAP: dict[str, Model]

    MODEL_SUMMARY_LIST: list[ModelSummary]
    MODEL_SUMMARY_MAP: dict[str, ModelSummary]

    META_LIST: list[Meta]
    META_MAP: dict[str, MetaNNExtractor]

    META_SUMMARY_LIST: list[MetaSummary]
    META_SUMMARY_MAP: dict[str, MetaNNExtractorSummary]

    IS_SERVING: bool


GLOBALS = Global(
    MODEL_LIST=[],
    MODEL_MAP={},

    MODEL_SUMMARY_LIST=[],
    MODEL_SUMMARY_MAP={},

    META_LIST=[],
    META_MAP={},

    IS_SERVING=False,
)

# logger default to root logger
logger: logging.Logger = logging.getLogger()

_NAME = 'nn-extractor'


def init(filename: str = '', extra_params: Optional[Config] = None):
    global config
    global logger

    version = _get_version()
    if extra_params is None:
        extra_params = {}
    extra_params['version'] = version

    logger, config = _cfg.init(_NAME, filename, default=config, extra_params=extra_params)

    if config['is_debug_config']:
        logger.debug(f'config: {config}')


def _get_version() -> str:
    return importlib.metadata.version('nn-extractor')


def check_disable(func: Callable, default: Any = None):
    def _check_disable(*args, **kwargs):
        if config['is_disable']:
            return default

        return func(*args, **kwargs)

    return _check_disable
