# -*- coding: utf-8 -*-

from typing import Optional


type Hook = tuple[str, function]

from .activation import Activation
from .record import Record

import torch
from torch.nn import Module

import logging


def register_hook(model: Module) -> tuple[list[Optional[Activation]], list[Optional[Hook]], list[Optional[Activation]], list[Optional[Hook]]]:
    activations, forward_hooks = register_forward_hook(model)
    gradients, backward_hooks = register_backward_hook(model)

    return activations, forward_hooks, gradients, backward_hooks


def register_forward_hook(model: Module):
    name_modules = model.named_modules()

    activations: list[Optional[Activation]] = [None] * len(name_modules)
    forward_hooks: list[Optional[Hook]] = [None] * len(name_modules)

    for idx, (name, module) in enumerate(model.named_modules()):
        forward_hooks[idx] = module.register_forward_hook(_get_forward_activation(idx, name, activations))

    return activations, forward_hooks


def register_backward_hook(model: Module):
    named_modules = model.named_modules()
    gradients: list[Optional[Activation]] = [None] * len(named_modules)
    backward_hooks: list[Optional[Hook]] = [None] * len(named_modules)

    for idx, (name, module) in enumerate(named_modules):
        backward_hooks[idx] = module.register_full_backward_hook(_get_backward_activation(idx, name, gradients))

    return gradients, backward_hooks


def _get_forward_activation(idx: int, name: str, activations: list[Optional[Activation]]):
    def hook(model: Module, args: tuple[torch.Tensor, ...], output: torch.Tensor):
        logging.debug(f'get_forward_activation ({idx}/{name})')
        sanitized_input = _sanitize_detach(args)
        params = [(each_name, _sanitize_detach(each)) for (each_name, each) in model.named_parameters(recurse=False)]
        sanitized_output = _sanitize_detach(output)
        activations[idx] = {'name': name, 'input': sanitized_input, 'params': params, 'output': sanitized_output}

        return hook


def _get_backward_activation(idx: int, name: str, gradients: list[Optional[Activation]]):

    def hook(model: Module, input: tuple[torch.Tensor, ...], output):
        logging.debug(f'get_backward_activation: ({idx}/{name})')
        sanitized_input = _sanitize_detach(input)
        params = [(each_name, _sanitize_detach(each)) for (each_name, each) in model.named_parameters(recurse=False)]
        sanitized_output = _sanitize_detach(output)
        gradients[idx] = {'name': name, 'input': sanitized_input, 'params': params, 'output': sanitized_output}

    return hook


def _sanitize_detach(val) -> Record:
    if isinstance(val, tuple):
        return (_sanitize_detach(each) for each in val)
    elif isinstance(val, list):
        return [_sanitize_detach(each) for each in val]
    elif isinstance(val, dict):
        return {k: _sanitize_detach(v) for k, v in val.items()}
    elif isinstance(val, torch.Tensor):
        return val.detach().to('cpu').numpy()
    else:
        return val
