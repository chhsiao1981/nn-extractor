# -*- coding: utf-8 -*-

from . import activation
from .activation import Activation
from . import nnextractor_pb2


def serialize_activations(activations: list[Activation]) -> nnextractor_pb2.Activations:
    activation_pbs = [activation.serialize_activation(each) for each in activations]
    return nnextractor_pb2.Activations(activations=activation_pbs)


def deserialize_activations(activations_pb: nnextractor_pb2.Activations) -> list[Activation]:
    activations: list[Activation] = list[None] * len(activations_pb.activations)
    for idx, each in enumerate(activations_pb.activations):
        activations[idx] = activation.deserialize_activation(each)
    return activations


def meta_activations(activations: list[Activation]) -> dict:
    the_len = len(activations)

    metadata = [activation.meta_activation(each) for each in activations]
    return {'len': the_len, 'activations': metadata}
