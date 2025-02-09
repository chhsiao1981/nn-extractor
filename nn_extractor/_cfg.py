# -*- coding: utf-8 -*-
# https://github.com/chhsiao1981/pyutil-cfg/blob/v0.1.1/pyutil_cfg/cfg.py

from enum import Enum
from typing import Optional, Any, TypedDict
try:
    from typing import NotRequired
except:  # noqa for 3.10
    from typing_extensions import NotRequired

from configparser import ConfigParser, RawConfigParser
try:
    import tomllib
except:  # noqa for 3.10
    import tomli as tomllib

import logging
import logging.config

import json


class FileType(Enum):
    ini = 1
    toml = 2

    def __str__(self):
        return self.name


class Loggers(TypedDict):
    keys: str
    disable_existing_loggers: NotRequired[bool]


class Logger(TypedDict):
    qualname: str
    handlers: str
    level: NotRequired[str]
    propagate: NotRequired[int | bool | str]


class Handlers(TypedDict):
    keys: str


Handler = TypedDict(
    'Handler',
    {
        'class': str,
        'args': NotRequired[str],
        'level': NotRequired[str],
        'formatter': NotRequired[str],
    }
)


class Formatters(TypedDict):
    keys: str


Formatter = TypedDict(
    'Formatter',
    {
        'format': NotRequired[str],
        'datefmt': NotRequired[str],
        'style': NotRequired[str],
        'validate': NotRequired[bool | str],
        'defaults': NotRequired[dict[str, float | int | bool | str]],
        'class': NotRequired[str],
    }
)


class Config(TypedDict):
    loggers: NotRequired[Loggers]
    handlers: NotRequired[Handlers]
    formatters: NotRequired[Formatters]

    '''
    logger_*: Logger
    '''

    '''
    handler_*: Handler
    '''

    '''
    formatter_*: Formatter
    '''


def init(
        name: str = '',
        filename: str = '',
        log_name: str = '',
        log_filename: str = '',
        extra_params: Optional[dict] = None,
        is_extra_params_in_file_ok: bool = True,
        is_skip_extra_params_in_file: bool = False,
        show_config: Optional[int] = None
) -> tuple[logging.Logger, Config]:
    '''
    init

    initialize the logger and config.

    Args:
        name: name to extract in the config file and as the logger name.
        filename: config filename. suffix with .toml or .ini

        log_name: opttionally log-specific name
        log_filename: optionally log-specific config filename.
        extra_params: optionally extra values not defined in the config filename.
                      usually from argparse.

        is_extra_params_in_file_ok:
            whether to raise error if extra_params already defined in the config file
            (as programmatically rewriting the settings from config file)
            default is ok.

        is_skip_extra_params_in_file:
            whether to skip extra if extra is in file.
            default tih False as extra can overwrite the config in file.

        show_config: log-level in the config that shows config before return.
                     None means not showing the config.

    Return:
        logger, config
    '''
    if log_name == '':
        log_name = name
    if log_filename == '':
        log_filename = filename

    # logger
    config_from_log_file = _config_from_file(log_filename)
    logger = _init_logger(log_name, config_from_log_file)

    # config
    config_from_file = _config_from_file(filename)
    config = _init_config(name, config_from_file)
    config = _post_init_config(
        config,
        extra_params,
        is_extra_params_in_file_ok,
        is_skip_extra_params_in_file,
        logger,
    )

    if show_config is not None:
        logger.log(show_config, f'pyutil_cfg.init: name: {name} logger-name: {log_name} config: {config}')   # noqa

    return logger, config


def _config_from_file(filename: str) -> tuple[Config]:
    if filename == '':
        return {}

    if filename.endswith('.toml'):
        config = _config_from_toml_file(filename)
        return config
    elif filename.endswith('.ini'):
        config = _config_from_ini_file(filename)
        return config
    else:
        raise Exception(f'not supported file type: filename: {filename}')


def _config_from_toml_file(filename: str) -> Config:
    with open(filename, 'rb') as f:
        return tomllib.load(f)


def _init_logger(name: str, config: Config) -> logging.Logger:
    logger = logging.getLogger(name)

    logger_config = {
        section: val
        for section, val
        in config.items()
        if _is_valid_logger_section(section)
    }

    if 'loggers' not in logger_config and 'formatters' not in logger_config and 'handlers' not in logger_config:  # noqa
        return logger

    logger_configparser = RawConfigParser()
    for section, val_by_section in logger_config.items():
        logger_configparser[section] = val_by_section

    # following logging.config.fileConfig default setting of disable_existing_loggers
    loggers: dict = logger_config.get('loggers', {})
    disable_existing_loggers = loggers.get('disable_existing_loggers', True)
    disable_existing_loggers = _val_to_json(disable_existing_loggers)

    try:
        logging.config.fileConfig(
            logger_configparser,
            disable_existing_loggers=disable_existing_loggers,
        )
    except Exception as e:
        logging.warning(f'unable to setup logger: e: {e}')

    return logger


def _init_config(name: str, config_from_file: Config) -> dict:
    '''
    setup config from config_from_file[name]
    '''
    return config_from_file.get(name, {})


def _config_from_ini_file(filename: str) -> dict[str, Any]:
    '''
    get ini conf from section
    return: config: {key: val} val: json_loaded
    '''
    if not filename:
        return {}

    config_parser = ConfigParser()
    config_parser.read(filename)
    sections = config_parser.sections()

    return {section: _parse_options(config_parser, section) for section in sections}


def _parse_options(config_parser: ConfigParser, section: str) -> dict[str, Any]:
    options = config_parser.options(section)
    return {option: _parse_option(option, section, config_parser) for option in options}


def _parse_option(option: str, section: str, config_parser: ConfigParser) -> Any:
    '''
    '''
    # special treatment to logger sections (get raw values)
    if _is_valid_logger_section(section):
        return config_parser.get(section, option, raw=True)

    # json-formatted values for other sections.
    val = config_parser.get(section, option)
    return _val_to_json(val)


def _val_to_json(val: Any) -> Any:
    '''
    try to do json load on value
    '''

    if not isinstance(val, str):
        return val

    orig_v = val
    try:
        val = json.loads(val)
    except:  # noqa
        val = orig_v

    return val


def _is_valid_logger_section(section: str) -> bool:
    if section in ['loggers', 'handlers', 'formatters']:
        return True

    if section.startswith('logger_'):
        return True
    if section.startswith('handler_'):
        return True
    if section.startswith('formatter_'):
        return True

    return False


def _post_init_config(
        config: dict,
        extra_params:
        Optional[dict],
        is_extra_params_in_file_ok: bool,
        is_skip_extra_params_in_file: bool,
        logger: logging.Logger,
) -> dict:
    '''
    add additional parameters into config
    '''

    if not extra_params:
        return config

    # warning if not is_extra_in_file_ok
    if not is_extra_params_in_file_ok:
        for k, v in extra_params.items():
            if k in config:
                logger.warning('config will be overwritten by extras: key: %s origin: %s new: %s', k, config[k], v)  # noqa

    # set valid extras, with is_skip_extra_in_file
    valid_extras = extra_params
    if is_skip_extra_params_in_file:
        valid_extras = {k: v for k, v in extra_params if k not in config}

    config.update(valid_extras)

    return config
