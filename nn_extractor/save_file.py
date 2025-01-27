# -*- coding: utf-8 -*-

from typing import Optional

import os
import math
import yaml

from . import constants
from . import activation
from .activation import Activation


def save_file(the_dir: str, activations: Optional[list[Activation]] = None, gradients: Optional[list[Activation]] = None, iteration: Optional[int] = None):
    full_dir = the_dir
    if iteration is not None:
        full_dir = os.sep.join([the_dir, f'{iteration}'])

    if activations is not None:
        _save_file(full_dir, constants.ACTIVATIONS, activations)

    if gradients is not None:
        _save_file(full_dir, constants.GRADIENTS, gradients)


def _save_file(the_dir: str, prefix: str, the_activations: list[Activation]):
    if len(the_activations) == 0:
        return

    full_dir = os.sep.join([the_dir, prefix])
    os.makedirs(full_dir, exist_ok=True)

    prefix_len = int(math.log10(len(the_activations))) + 1

    prefix_len_str = f'%0{prefix_len}d'

    metadata = {
        'len': len(the_activations),
    }
    full_filename = os.sep.join([full_dir, constants.METADATA_FILENAME])
    with open(full_filename, 'w') as f:
        yaml.dump_all(metadata, f)

    for idx, each_activation in enumerate(the_activations):
        activation_pb = activation.serialize_activations(each_activation)

        filename = f'{prefix_len_str % (idx)}_{each_activation.name}.pb'
        full_filename = os.sep.join([full_dir, filename])
        with open(full_filename, 'wb') as f:
            f.write(activation_pb.SerializeToString())
