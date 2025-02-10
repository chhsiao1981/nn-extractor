# -*- coding: utf-8 -*-

from .types import Model, ModelSummary

from nn_extractor.examples.nnunetextractor import nnUNetPredictor

from nn_extractor import cfg


def init_model():
    nnunetv2_predict_model: Model = {
        'name': 'nnUNetv2',
        'classname': 'nn_extractor.cmds.nnunetextractor.nnUNetPredictor',
        'default_params': {
            'tile_step_size': 0.5,
            'use_gaussian': True,
            'use_mirroring': True,
            'perform_everything_on_device': True,
            'device': 'cuda',
            'allow_tqdm': False,
        },
        'the_class': nnUNetPredictor,
    }

    model_list: list[Model] = [nnunetv2_predict_model]
    model_map = {each['name']: each for each in model_list}

    model_summary_list = list(map(lambda each: _model_summary(each), model_list))
    model_summary_map = {each['name']: each for each in model_summary_list}

    cfg.logger.info(f'loaded model: ({model_map.keys()} / {len(model_map)}')

    cfg.GLOBALS['MODEL_LIST'] = model_list
    cfg.GLOBALS['MODEL_MAP'] = model_map
    cfg.GLOBALS['MODEL_SUMMARY_LIST'] = model_summary_list
    cfg.GLOBALS['MODEL_SUMMARY_MAP'] = model_summary_map


def _model_summary(model: Model) -> ModelSummary:
    return {k: v for k, v in model.items() if k != 'the_class'}
