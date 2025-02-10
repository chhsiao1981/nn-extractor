# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, Optional, Any, Callable, NamedTuple

from nn_extractor.nntensor import NNTensorType
from nn_extractor.ops.spacing import Spacing
import numpy as np
import os

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

from . import cfg
from .nnrecord import NNRecord, NNRecordMeta
from .nnparameter import NNParameter
from . import nnextractor_pb2

from . import profile
from . import nntensor

from .types import MetaNNNode, MetaNNRecord, MetaNNParameter, RecursiveNNTensor


class Hook(NamedTuple):
    name: str
    hook: Callable


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

    '''
    to let the root node knows all children.
    '''
    all_children: Optional[list[Self]] = None

    the_dir: str = ''
    the_index: int = 0
    data_id: str = ''

    _is_init_children: bool = False

    def __init__(
        self: Self,
        name: str,

        model: Optional[Module] = None,

        the_dir: str = '',
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

        if not the_dir:
            the_dir = '.'

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

        self.the_dir = the_dir
        self.the_index = index

        # the_dir already includes the index.
        self.data_id = os.sep.join([the_dir, self.name])

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

    def forward_serialize_pb(
        self,
        is_serialize_inputs=True,
        is_serialize_children=True,
    ) -> nnextractor_pb2.NNNode:

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

    def backward_serialize_pb(
        self,
        is_serialize_inputs=True,
        is_serialize_children=True,
    ) -> nnextractor_pb2.NNNode:
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

    def serialize_pb(
        self,
        is_serialize_inputs=True,
        is_serialize_children=True,
    ) -> nnextractor_pb2.NNNode:
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
        activation = None \
            if nnnode_pb.activation.the_type == nnextractor_pb2.NNRecordType.NNR_UNSPECIFIED \
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
        '''
        meta

        assuming that the meta of the nnnode is immuntable.
        '''
        inputs = self._get_meta_nnrecords(self.inputs)
        params = self._get_meta_params(self.params)
        activation = self._get_meta_nnrecord(self.activation)

        gradient_inputs = self._get_meta_nnrecords(self.gradient_inputs)
        gradient_params = self._get_meta_params(self.gradient_params)
        gradients = self._get_meta_nnrecords(self.gradients)

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

    def _get_meta_nnrecords(
        self: Self,
        nnrecord: Optional[list[NNRecord]],
    ) -> Optional[list[MetaNNRecord]]:
        if nnrecord is None:
            return None

        meta_nnrecords = [each.meta() for each in nnrecord]
        return meta_nnrecords

    def _get_meta_params(
        self: Self,
        params: Optional[list[NNParameter]],
    ) -> Optional[list[MetaNNParameter]]:
        if params is None:
            return None

        meta_params = [each.meta() for each in params]
        return meta_params

    def _get_meta_nnrecord(self: Self, nnrecord: Optional[NNRecord]) -> MetaNNRecord:
        if nnrecord is None:
            return None

        meta = nnrecord.meta()
        return meta

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
            each_the_dir = os.sep.join([self.the_dir, str(idx)])
            each_children = NNNode(
                name=full_name,
                model=each_module,
                the_dir=each_the_dir,
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
        forward_hook = self.model.register_forward_hook(
            self._get_forward_activation(),
            with_kwargs=True,
        )
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
                if args:
                    sanitized_args = [
                        NNRecord(self._sanitize_detach(each), data_id=f'{self.data_id}/args/{idx}')
                        for idx, each in enumerate(args)
                    ]
                if kwargs:
                    sanitized_kwargs = [
                        NNRecord(
                            self._sanitize_detach(each),
                            name=each_name,
                            data_id=f'{self.data_id}/kwargs/{each_name}',
                        )
                        for each_name, each in kwargs.items()]
            else:
                if args:
                    sanitized_args = [
                        NNRecord(self._sanitize_meta(each), data_id=f'{self.data_id}/args/{idx}')
                        for idx, each in enumerate(args)
                    ]
                if kwargs:
                    sanitized_kwargs = [
                        NNRecord(
                            self._sanitize_meta(each),
                            name=each_name,
                            data_id=f'{self.data_id}/kwargs/{each_name}',
                        )
                        for each_name, each in kwargs.items()]

            params = [
                NNParameter(
                    name=each_name,
                    parameter=self._sanitize_detach(each, is_op_scale=False),
                    data_id=f'{self.data_id}/params/{each_name}',
                )
                for (each_name, each)
                in model.named_parameters(recurse=False)]

            sanitized_output = NNRecord(self._sanitize_detach(
                output), data_id=f'{self.data_id}/activation')

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
        backward_hook = self.model.register_full_backward_hook(
            self._get_backward_activation(), with_kwargs=True)
        self.backward_hook = backward_hook

        for each_children in self.children:
            each_children.register_backward_hook()

    def _get_backward_activation(self: Self):
        def hook(
            model: Module,
            grad_inputs: Optional[tuple[torch.Tensor, ...]],
            gradients: Optional[tuple[torch.Tensor, ...]],
        ):
            sanitized_inputs = []
            if grad_inputs:
                sanitized_inputs = [
                    NNRecord(
                        value=self._sanitize_detach(each),
                        data_id=f'{self.data_id}/grad_inputs/{idx}',
                    )
                    for idx, each in enumerate(grad_inputs)]

            params = [
                NNParameter(
                    name=each_name,
                    parameter=self._sanitize_detach(each, is_op_scale=False),
                    data_id=f'{self.data_id}/grad_params/{each_name}',
                )
                for (each_name, each)
                in model.named_parameters(recurse=False)]

            sanitized_gradients = []
            if gradients:
                sanitized_gradients = [
                    NNRecord(
                        value=self._sanitize_detach(each),
                        data_id=f'{self.data_id}/gradients/{idx}',
                    )
                    for idx, each in enumerate(gradients)]

            self.gradient_inputs = sanitized_inputs
            self.gradient_params = params
            self.gradients = sanitized_gradients

        return hook

    def remove_backward_hook(self: Self):
        if self.backward_hook is None:
            return

        self.backward_hook.remove()
        self.backward_hook = None

    def _sanitize_detach(self: Self, val: RecursiveNNTensor, is_op_scale=True) -> RecursiveNNTensor:
        if isinstance(val, tuple) or isinstance(val, list):
            return [self._sanitize_detach(each, is_op_scale) for each in val]
        elif isinstance(val, dict):
            return {k: self._sanitize_detach(v, is_op_scale) for k, v in val.items()}
        elif isinstance(val, torch.Tensor):
            ret = val.detach().to('cpu').numpy()
            if not is_op_scale:
                return ret
            return Spacing(ret).integrate(self.name)
        elif isinstance(val, np.ndarray):
            if not is_op_scale:
                return val
            return Spacing(val).integrate(self.name)
        else:
            cfg.logger.warning(f'NNNode._sanitize_detach: unknown val type: {type(val)}')
            return np.array([])

    def _sanitize_meta(self: Self, val: RecursiveNNTensor) -> NNRecordMeta:
        if isinstance(val, tuple) or isinstance(val, list):
            return [self._sanitize_meta(each) for each in val]
        elif isinstance(val, dict):
            return {k: self._sanitize_meta(v) for k, v in val.items()}
        elif isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            return NNRecordMeta(shape=val.shape, the_type=nntensor.dtype(val))
        elif isinstance(val, NNRecordMeta):
            return val
        else:
            cfg.logger.warning(f'NNNode._sanitize_detach: unknown val type: {val} ({type(val)})')
            return NNRecordMeta(shape=tuple([]), the_type=NNTensorType.UNSPECIFIED)

    def forward_save_to_file(self: Self, seq_dir: str):
        if not self.all_children:
            return

        for each in tqdm(self.all_children, desc=f'({seq_dir}) node: {self.name}'):
            each._each_forward_save_to_file(seq_dir)

    def _each_forward_save_to_file(self: Self, seq_dir: str):
        if self.inputs is not None and cfg.config['is_nnnode_record_inputs']:
            [each.save_to_file(seq_dir) for each in self.inputs]
        if self.params is not None:
            [each.save_to_file(seq_dir) for each in self.params]
        if self.activation is not None:
            self.activation.save_to_file(seq_dir)

    def backward_save_to_file(self: Self, seq_dir: str):
        if not self.all_children:
            return

        for each in tqdm(self.all_children, desc=f'({seq_dir}) node: {self.name}'):
            each._each_backward_save_to_file(seq_dir)

    def _each_backward_save_to_file(self: Self, seq_dir: str):
        if self.gradient_inputs is not None and cfg.config['is_nnnode_record_inputs']:
            [each.save_to_file(seq_dir) for each in self.gradient_inputs]
        if self.gradient_params is not None:
            [each.save_to_file(seq_dir) for each in self.gradient_params]
        if self.gradients is not None:
            [each.save_to_file(seq_dir) for each in self.gradients]
