# -*- coding: utf-8 -*-

from nn_extractor import cfg


def get_model_list_handler(start_idx: int, n: int, is_asc: bool):
    '''
    '''
    models = cfg.GLOBALS['MODEL_SUMMARY_LIST']

    end_idx = start_idx + n
    if not is_asc:
        end_idx = start_idx
        start_idx = max(end_idx - n, 0)

    return models[start_idx:end_idx]
