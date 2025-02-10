# -*- coding: utf-8 -*-

from nn_extractor import cfg

from nn_extractor.server.types import Meta


def get_meta_handler(meta_id: str) -> Meta:
    cfg.logger.info(f'meta_id: {meta_id}')
    return cfg.GLOBALS['META_MAP'].get(meta_id, {})
