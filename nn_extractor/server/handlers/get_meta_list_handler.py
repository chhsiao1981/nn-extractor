# -*- coding: utf-8 -*-

from nn_extractor import cfg


def get_meta_list_handler(start_idx: int, n: int, is_asc: bool):
    '''
    get_meta_list_hanlder

    get the list of root-extractor meta-names:
    '''

    meta_list = cfg.GLOBALS['META_SUMMARY_LIST']

    end_idx = start_idx + n
    if not is_asc:
        end_idx = start_idx
        start_idx = max(end_idx - n, 0)

    return meta_list[start_idx:end_idx]
