# -*- coding: utf-8 -*-

from typing import TypedDict

from . import record
from .record import Record

from . import parameter
from .parameter import Parameter

from . import nnextractor_pb2

import logging


class Activation(TypedDict):
    name: str
    input: Record
    params: list[Parameter]
    output: Record


def is_same_activation(activation_a: Activation, activation_b: Activation):
    if not _is_valid_type(activation_a):
        return False

    if not _is_valid_type(activation_b):
        return False

    if activation_a['name'] != activation_b['name']:
        return False

    if not record.is_same_record(activation_a['input'], activation_b['input']):
        return False

    a_params = activation_a['params']
    b_params = activation_b['params']
    if len(a_params) != len(b_params):
        return False

    for idx, each in enumerate(a_params):
        if not parameter.is_same_parameter(each, b_params[idx]):
            return False

    if not record.is_same_record(activation_a['output'], activation_b['output']):
        return False

    return True


def _is_valid_type(act: Activation) -> bool:
    if not isinstance(act, dict):
        return False

    if 'name' not in act:
        return False

    if 'input' not in act:
        return False

    if 'params' not in act:
        return False

    if not isinstance(act['params'], list):
        return False

    if 'output' not in act:
        return False

    return True


def serialize_activation(the_activation: Activation) -> nnextractor_pb2.Activation:
    input_pb = record.serialize_record(the_activation['input'])
    params_pb = [parameter.serialize_parameter(each) for each in the_activation['params']]
    output_pb = record.serialize_record(the_activation['output'])

    return nnextractor_pb2.Activation(
        name=the_activation['name'],
        input=input_pb,
        params=params_pb,
        output=output_pb,
    )


def deserialize_activation(activation_pb: nnextractor_pb2.Activation) -> Activation:
    input = record.deserialize_record(activation_pb.input)
    params: list[Parameter] = [None] * len(activation_pb.params)
    for idx, each in enumerate(activation_pb.params):
        params[idx] = parameter.deserialize_parameter(each)
    output = record.deserialize_record(activation_pb.output)

    return Activation(name=activation_pb.name, input=input, params=params, output=output)


def meta_activation(activation: Activation) -> dict:
    meta_input = record.meta_record(activation.input)
    meta_params = [parameter.meta_parameter(each) for each in activation.params]
    meta_output = record.meta_record(activation.output)

    return {
        'name': activation.name,
        'input': meta_input,
        'params': meta_params,
        'output': meta_output,
    }
