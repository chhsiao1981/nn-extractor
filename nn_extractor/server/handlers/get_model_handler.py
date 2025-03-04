# -*- coding: utf-8 -*-

from nn_extractor import cfg

from nn_extractor.server.types import ModelSummary


def get_model_handler(model_id: str) -> ModelSummary:
    return cfg.GLOBALS.get('MODEL_SUMMARY_MAP', {}).get(model_id, {})
