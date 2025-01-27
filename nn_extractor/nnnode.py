# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, TypedDict, Optional, Any, Callable, NamedTuple

import numpy as np
import os
import pickle

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from . import cfg
from .ndarray import NDArray
from .nnrecord import NNRecord, MetaNNRecord
from .nnparameter import NNParameter, MetaNNParameter
from . import nnextractor_pb2

from . import profile

type Tensor = torch.Tensor | tuple[Tensor] | list[Tensor] | dict[str, Tensor]


class Hook(NamedTuple):
    name: str
    hook: Callable


class MetaNNNode(TypedDict):
    name: str

    inputs: list[MetaNNRecord]
    params: list[MetaNNParameter]
    activation: Optional[MetaNNRecord]

    gradient_inputs: list[MetaNNRecord]
    gradient_params: list[MetaNNParameter]
    gradients: list[MetaNNRecord]

    children: Optional[list[Self]]


class PickleNNNode(NamedTuple):
    name: str
    inputs: Optional[list[NNRecord]] = None
    params: Optional[list[NNParameter]] = None
    activation: Optional[NNRecord] = None

    gradient_inputs: Optional[list[NNRecord]] = None
    gradient_params: Optional[list[NNParameter]] = None
    gradients: Optional[list[NNRecord]] = None

    children: Optional[list[Self]] = None


@dataclass
class NNNode(object):
    name: str

    model: Optional[Module] = None

    inputs: Optional[list[NNRecord]] = None
    params: Optional[list[NNParameter]] = None
    activation: Optional[NNRecord] = None
    forward_hook: Optional[RemovableHandle] = None

    gradient_inputs: Optional[list[NNRecord]] = None
    gradient_params: Optional[list[NNParameter]] = None
    gradients: Optional[list[NNRecord]] = None
    backward_hook: Optional[RemovableHandle] = None

    children: Optional[list[Self]] = None

    all_children: Optional[list[Self]] = None

    the_dir: str = ''
    the_index: int = 0
    filename: str = ''

    _is_init_children: bool = False

    def __init__(
        self: Self,
        name: str,

        model: Optional[Module] = None,

        output_dir: str = '',
        index: int = 0,

        inputs: Optional[list[NNRecord]] = None,
        params: Optional[list[NNParameter]] = None,
        activation: Optional[NNRecord] = None,

        gradient_inputs: Optional[list[NNRecord]] = None,
        gradient_params: Optional[list[NNParameter]] = None,
        gradients: Optional[list[NNRecord]] = None,

        children: Optional[list[Self]] = None,

        all_children: Optional[list[Self]] = None,
    ):
        '''
        2 modes in initialization:

        1. init with deserialization (model is None)
        2. init with model (model is not None)
        '''

        if inputs is None:
            inputs = []
        if params is None:
            params = []

        if gradient_inputs is None:
            gradient_inputs = []

        if gradient_params is None:
            gradient_params = []

        if gradients is None:
            gradients = []

        if children is None:
            children = []

        if not output_dir:
            output_dir = '.'

        self.name = name
        self.inputs = inputs
        self.params = params
        self.activation = activation
        self.gradient_inputs = gradient_inputs
        self.gradient_params = gradient_params
        self.gradients = gradients
        self.children = children

        if model is None:
            # method 1: init with deserialization
            return

        # method 2: init with model
        self.model = model

        self.the_dir = output_dir
        self.the_index = index
        self.filename = f"{'_'.join(list(filter(None, [self.name, str(index)])))}"

        self.init_children(all_children)

    def __eq__(self: Self, value: Any) -> bool:
        if not isinstance(value, NNNode):
            return False

        if self.name != value.name:
            return False
        if self.inputs != value.inputs:
            return False
        if self.params != value.params:
            return False
        if self.activation != value.activation:
            return False

        if self.gradient_inputs != value.gradient_inputs:
            return False
        if self.gradient_params != value.gradient_params:
            return False
        if self.gradients != value.gradients:
            return False

        if self.children != value.children:
            return False
        return True

    def forward_serialize_pk(self, is_serialize_inputs=True, is_serialize_children=True) -> PickleNNNode:
        profile.profile_start('nnnode.forward_serialize_pk')
        serialized_inputs = []
        if is_serialize_inputs and self.inputs:
            serialized_inputs = [each.serialize_pk() for each in self.inputs]

        serialized_params = [] if self.params is None \
            else [each.serialize_pk() for each in self.params]
        serialized_activation = None if self.activation is None \
            else self.activation.serialize_pk()
        profile.profile_stop('nnnode.forward_serialize_pk')

        serialized_children = []
        if is_serialize_children and self.children:
            serialized_children = [each.forward_serialize_pk() for each in self.children]

        return PickleNNNode(
            name=self.name,
            inputs=serialized_inputs,
            params=serialized_params,
            activation=serialized_activation,
            children=serialized_children,
        )

    @classmethod
    def deserialize_pk(cls, nnnode_pk: PickleNNNode) -> Self:
        inputs = None if nnnode_pk.inputs is None \
            else [NNRecord.deserialize_pk(each) for each in nnnode_pk.inputs]
        params = None if nnnode_pk.params is None \
            else [NNParameter.deserialize_pk(each) for each in nnnode_pk.params]
        activation = None if nnnode_pk.activation is None \
            else NNRecord.deserialize_pk(nnnode_pk.activation)

        gradient_inputs = None if nnnode_pk.gradient_inputs is None \
            else [NNRecord.deserialize_pk(each) for each in nnnode_pk.gradient_inputs]
        gradient_params = None if nnnode_pk.gradient_params is None \
            else [NNParameter.deserialize_pk(each) for each in nnnode_pk.params]
        gradients = None if nnnode_pk.gradients is None \
            else [NNRecord.deserialize_pk(each) for each in nnnode_pk.gradients]
        children = None if nnnode_pk.children is None \
            else [NNNode.deserialize_pk(each) for each in nnnode_pk.children]
        return NNNode(
            name=nnnode_pk.name,

            inputs=inputs,
            params=params,
            activation=activation,

            gradient_inputs=gradient_inputs,
            gradient_params=gradient_params,
            gradients=gradients,

            children=children,
        )

    def forward_serialize_pb(self, is_serialize_inputs=True, is_serialize_children=True) -> nnextractor_pb2.NNNode:

        profile.profile_start('nnnode.forward_serialize_pb')
        serialized_inputs = []
        if is_serialize_inputs and self.inputs:
            serialized_inputs = [each.serialize_pb() for each in self.inputs]

        serialized_params = [] if self.params is None \
            else [each.serialize_pb() for each in self.params]
        serialized_activation = None if self.activation is None \
            else self.activation.serialize_pb()
        profile.profile_stop('nnnode.forward_serialize_pb')

        serialized_children = []
        if is_serialize_children and self.children:
            serialized_children = [each.forward_serialize_pb() for each in self.children]

        return nnextractor_pb2.NNNode(
            name=self.name,
            inputs=serialized_inputs,
            params=serialized_params,
            activation=serialized_activation,
            children=serialized_children,
        )

    def backward_serialize_pb(self, is_serialize_inputs=True, is_serialize_children=True) -> nnextractor_pb2.NNNode:
        serialized_gradient_inputs = []
        if is_serialize_inputs and self.gradient_inputs is not None:
            serialized_gradient_inputs = [each.serialize_pb() for each in self.gradient_inputs]

        serialized_gradient_params = [] if self.gradient_params is None \
            else [each.serialize_pb() for each in self.gradient_params]
        serialized_gradients = [] if self.gradients is None \
            else [each.serialize_pb() for each in self.gradients]

        serialized_children = []
        if is_serialize_children:
            serialized_children = [each.backward_serialize_pb() for each in self.children]

        return nnextractor_pb2.NNNode(
            name=self.name,
            gradient_inputs=serialized_gradient_inputs,
            gradient_params=serialized_gradient_params,
            gradients=serialized_gradients,
            children=serialized_children,
        )

    def serialize_pb(self, is_serialize_inputs=True, is_serialize_children=True) -> nnextractor_pb2.NNNode:
        serialized_inputs = []
        if is_serialize_inputs:
            serialized_inputs = [each.serialize_pb() for each in self.inputs]

        serialized_params = [] if self.params is None \
            else [each.serialize_pb() for each in self.params]
        serialized_activation = None if self.activation is None \
            else self.activation.serialize_pb()

        serialized_gradient_inputs = []
        if is_serialize_inputs and self.gradient_inputs is not None:
            serialized_gradient_inputs = [each.serialize_pb() for each in self.gradient_inputs]

        serialized_gradient_params = [] if self.gradient_params is None \
            else [each.serialize_pb() for each in self.gradient_params]
        serialized_gradients = [] if self.gradients is None \
            else [each.serialize_pb() for each in self.gradients]

        serialized_children = []
        if is_serialize_children:
            serialized_children = [each.backward_serialize_pb() for each in self.children]

        return nnextractor_pb2.NNNode(
            name=self.name,
            inputs=serialized_inputs,
            params=serialized_params,
            activation=serialized_activation,
            gradient_inputs=serialized_gradient_inputs,
            gradient_params=serialized_gradient_params,
            gradients=serialized_gradients,
            children=serialized_children,
        )

    @classmethod
    def deserialize_pb(cls, nnnode_pb: nnextractor_pb2.NNNode) -> Self:
        inputs = [NNRecord.deserialize_pb(each) for each in nnnode_pb.inputs]
        params = [NNParameter.deserialize_pb(each) for each in nnnode_pb.params]
        activation = None if nnnode_pb.activation.the_type == nnextractor_pb2.NNRecordType.NNR_UNSPECIFIED \
            else NNRecord.deserialize_pb(nnnode_pb.activation)

        gradient_inputs = [NNRecord.deserialize_pb(each) for each in nnnode_pb.gradient_inputs]
        gradient_params = [NNParameter.deserialize_pb(each) for each in nnnode_pb.params]
        gradients = [NNRecord.deserialize_pb(each) for each in nnnode_pb.gradients]
        children = [NNNode.deserialize_pb(each) for each in nnnode_pb.children]
        return NNNode(
            name=nnnode_pb.name,

            inputs=inputs,
            params=params,
            activation=activation,

            gradient_inputs=gradient_inputs,
            gradient_params=gradient_params,
            gradients=gradients,

            children=children,
        )

    def meta(self) -> MetaNNNode:
        inputs = [each.meta() for each in self.inputs]
        params = [each.meta() for each in self.params]
        activation = None if self.activation is None else self.activation.meta()

        gradient_inputs = [each.meta() for each in self.gradient_inputs]
        gradient_params = [each.meta() for each in self.gradient_params]
        gradients = [each.meta() for each in self.gradients]
        children = None if self.children is None else [each.meta() for each in self.children]

        return MetaNNNode(
            name=self.name,
            inputs=inputs,
            params=params,
            activation=activation,

            gradient_inputs=gradient_inputs,
            gradient_params=gradient_params,
            gradients=gradients,

            children=children,
        )

    def init_children(self: Self, all_children: Optional[list[Self]] = None):
        if self.model is None:
            return

        if self._is_init_children:
            return

        is_to_set_all_children = False
        if all_children is None:
            is_to_set_all_children = True
            all_children = [self]
        else:
            all_children.append(self)

        named_children = list(self.model.named_children())
        children: list[NNNode] = [None] * len(named_children)
        for idx, (each_name, each_module) in enumerate(named_children):
            full_name = f'{self.name}.{each_name}'
            each_output_dir = os.sep.join([self.the_dir, str(idx)])
            each_children = NNNode(
                name=full_name,
                model=each_module,
                output_dir=each_output_dir,
                index=idx,
                all_children=all_children,
            )
            children[idx] = each_children

        self.children = children

        if is_to_set_all_children:
            self.all_children = all_children

        self._is_init_children = True

    def register_forward_hook(self: Self):
        # already registered forward hook
        if self.forward_hook is not None:
            return None

        # no model specified yet
        if self.model is None:
            return

        # register forward hook
        forward_hook = self.model.register_forward_hook(self._get_forward_activation(), with_kwargs=True)
        self.forward_hook = forward_hook

        for each_children in self.children:
            each_children.register_forward_hook()

    def _get_forward_activation(self: Self):
        def hook(
            model: Module,
            args: Optional[tuple[torch.Tensor, ...]],
            kwargs: Optional[dict[str, torch.Tensor]],
            output: torch.Tensor,
        ):

            sanitized_args = []
            sanitized_kwargs = []
            if cfg.config['is_nnnode_record_inputs']:
                sanitized_args = [] if not args \
                    else [NNRecord(self._sanitize_detach(each)) for each in args]
                sanitized_kwargs = [] if not kwargs \
                    else [NNRecord(self._sanitize_detach(each), name=each_name) for each_name, each in kwargs.items()]

            params = [NNParameter(name=each_name, record=self._sanitize_detach(each))
                      for (each_name, each)
                      in model.named_parameters(recurse=False)]

            sanitized_output = NNRecord(self._sanitize_detach(output))

            self.inputs = sanitized_args + sanitized_kwargs
            self.params = params
            self.activation = sanitized_output

        return hook

    def remove_forward_hook(self: Self):
        if self.forward_hook is None:
            return

        self.forward_hook.remove()
        self.forward_hook = None

    def register_backward_hook(self: Self):
        # already registered backward hook
        if self.backward_hook is not None:
            return

        # no model specified yet.
        if self.model is None:
            return

        # register backward hook
        backward_hook = self.model.register_full_backward_hook(self._get_backward_activation(), with_kwargs=True)
        self.backward_hook = backward_hook

        for each_children in self.children:
            each_children.register_backward_hook()

    def _get_backward_activation(self: Self):
        def hook(
            model: Module,
            grad_inputs: Optional[tuple[torch.Tensor, ...]],
            grad_outputs: Optional[tuple[torch.Tensor, ...]],
        ):
            sanitized_inputs = [] if not grad_inputs \
                else [NNRecord(self._sanitize_detach(each)) for each in grad_inputs]
            params = [NNParameter(name=each_name, record=NNRecord(self._sanitize_detach(each)))
                      for (each_name, each)
                      in model.named_parameters(recurse=False)]
            sanitized_outputs = [] if not grad_outputs \
                else [NNRecord(self._sanitize_detach(each)) for each in grad_outputs]

            self.gradient_inputs = sanitized_inputs
            self.gradient_params = params
            self.gradients = sanitized_outputs

        return hook

    def remove_backward_hook(self: Self):
        if self.backward_hook is None:
            return

        self.backward_hook.remove()
        self.backward_hook = None

    def _sanitize_detach(self: Self, val: Tensor) -> NDArray:
        if isinstance(val, tuple):
            return [self._sanitize_detach(each) for each in val]
        elif isinstance(val, list):
            return [self._sanitize_detach(each) for each in val]
        elif isinstance(val, dict):
            return {k: self._sanitize_detach(v) for k, v in val.items()}
        elif isinstance(val, torch.Tensor):
            return val.detach().to('cpu').numpy()
        else:
            return np.array([])

    def forward_save_to_file(self: Self, output_dir: str, index: int):
        if not self.all_children:
            return

        is_serialize_inputs = cfg.config['is_nnnode_record_inputs']

        full_output_dir = os.sep.join([output_dir, str(index)])
        for each in tqdm(self.all_children, desc=f'({index}) node: {self.name}'):
            # each: Self  # XXX for typing
            each._each_forward_save_to_file_pb(full_output_dir, is_serialize_inputs)

    def _each_forward_save_to_file_pk(self: Self, root_dir: str, is_serialize_inputs: bool):
        serialized = self.forward_serialize_pk(is_serialize_inputs=is_serialize_inputs, is_serialize_children=False)

        profile.profile_start('nnnode._each_forward_save_to_file_pk')

        the_dir = os.sep.join([root_dir, self.the_dir])
        if not os.path.exists(the_dir):
            os.makedirs(the_dir, exist_ok=True)
        out_filename = f"{os.sep.join([the_dir, self.filename])}.pk"
        with open(out_filename, 'wb') as f:
            pickle.dump(serialized, f)

        profile.profile_stop('nnnode._each_forward_save_to_file_pk')

    def _each_forward_save_to_file_pb(self: Self, root_dir: str, is_serialize_inputs: bool):
        serialized = self.forward_serialize_pb(is_serialize_inputs=is_serialize_inputs, is_serialize_children=False)

        profile.profile_start('nnnode._each_forward_save_to_file_pb')

        the_dir = os.sep.join([root_dir, self.the_dir])
        if not os.path.exists(the_dir):
            os.makedirs(the_dir, exist_ok=True)
        out_filename = f"{os.sep.join([the_dir, self.filename])}.pb"
        with open(out_filename, 'wb') as f:
            f.write(serialized.SerializeToString())

        profile.profile_stop('nnnode._each_forward_save_to_file_pb')
