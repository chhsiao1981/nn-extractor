# -*- coding: utf-8 -*-

import os
import os.path
import sys

from nn_extractor import cfg
from nn_extractor import argparse
import traceback

from pydantic import BaseModel, Field

from . import meta
from . import model


class Args(BaseModel):
    ini: str = Field(
        alias='i',
        default='config.dev.toml',
        description='config file')


def parse_args():
    print('parse_pargs: start: sys.argv:', sys.argv)

    parser = argparse.ArgumentParser(
        description='Use this to run inference with nnU-Net. This function is used when '
        'you want to manually specify a folder containing a trained nnU-Net '
        'model. This is useful when the nnunet environment variables '
        '(nnUNet_results) are not set.')
    argparse.add_args(parser, Args)

    prog_basename = os.path.basename(sys.argv[0])
    args = sys.argv[3:] if prog_basename == 'fastapi' else sys.argv[1:]

    return parser.parse_args(args)


def init():
    args = parse_args()

    cfg.init(args.ini, extra_params={'is_debug_config': True})

    cfg.GLOBALS['IS_SERVING'] = True

    try:
        model.init_model()
        meta.init_meta()
    except Exception as e:
        cfg.logger.error(f'unable to init: e: {e}')
        traceback.print_exception(e)
        raise e
