# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, TypedDict, Optional

import json
from torch.nn import Module

import os

from . import cfg
from . import constants
from .items import Items, MetaItems

from .nnnode import NNNode, MetaNNNode

from .sequence import Sequence, SequenceType, MetaSequence


class MetaNNExtractor(TypedDict):
    name: str

    seq: list[MetaSequence]
    inputs: list[MetaItems]
    preprocesses: list[MetaItems]
    nodes: list[MetaNNNode]
    postprocesses: list[MetaItems]
    outputs: list[MetaItems]

    extractors: list[Self]


@dataclass
class NNExtractor(object):
    '''
    NNExtractor

    reqiring the nodes to be unique.
    '''
    name: str

    seq: list[Sequence]
    inputs: list[Items]
    preprocesses: list[Items]
    nodes: list[NNNode]
    postprocesses: list[Items]
    outputs: list[Items]

    nodes_by_name: dict[str, NNNode]

    extractors: list[Self]

    root_dir: str
    inputs_dir: str
    preprocess_dir: str
    forward_dir: str
    backward_dir: str
    postprocess_dir: str
    outputs_dir: str

    current_forward_node: Optional[NNNode] = None
    current_backward_node: Optional[NNNode] = None

    forward_snapshot_index: int = 0
    backward_snapshot_index: int = 0

    def __init__(self, name: str):
        self.name = name
        self.seq = []
        self.inputs = []
        self.preprocesses = []
        self.nodes = []
        self.postprocesses = []
        self.outputs = []

        self.nodes_by_name = {}
        self.extractors = []

        self.root_dir = os.sep.join([cfg.config['output_dir'], name])
        os.makedirs(self.root_dir, exist_ok=True)
        self.inputs_dir = os.sep.join([self.root_dir, constants.DIR_INPUTS])
        os.makedirs(self.inputs_dir, exist_ok=True)
        self.preprocess_dir = os.sep.join([self.root_dir, constants.DIR_PREPROCESSES])
        os.makedirs(self.preprocess_dir, exist_ok=True)
        self.forward_dir = os.sep.join([self.root_dir, constants.DIR_FORWARDS])
        os.makedirs(self.forward_dir, exist_ok=True)
        self.backward_dir = os.sep.join([self.root_dir, constants.DIR_BACKWARDS])
        os.makedirs(self.backward_dir, exist_ok=True)
        self.postprocess_dir = os.sep.join([self.root_dir, constants.DIR_POSTPROCESSES])
        os.makedirs(self.postprocess_dir, exist_ok=True)
        self.outputs_dir = os.sep.join([self.root_dir, constants.DIR_OUTPUTS])
        os.makedirs(self.outputs_dir, exist_ok=True)

    @cfg.check_disable
    def add_inputs(self: Self, data: dict, name: str = '', is_save=True):
        items = Items(name=name, item_dict=data)
        index = len(self.inputs)
        self.seq.append(Sequence(SequenceType.INPUTS, index))
        if is_save:
            output_dir = os.sep.join([self.inputs_dir, str(index)])
            items.save_to_file(output_dir)
        self.inputs.append(items)

    @cfg.check_disable
    def add_preprocess(self: Self, data: dict, name: str = '', is_save=True):
        items = Items(name=name, item_dict=data)
        index = len(self.preprocesses)
        self.seq.append(Sequence(SequenceType.PREPROCESS, index))
        if is_save:
            output_dir = os.sep.join([self.preprocess_dir, str(index)])
            items.save_to_file(output_dir)
        self.preprocesses.append(items)

    @cfg.check_disable
    def add_postprocess(self: Self, data: dict, name: str = '', is_save=True):
        items = Items(item_dict=data, name=name)
        index = len(self.postprocesses)
        self.seq.append(Sequence(SequenceType.POSTPROCESSES, index))
        if is_save:
            output_dir = os.sep.join([self.postprocess_dir, str(index)])
            items.save_to_file(output_dir)
        self.postprocesses.append(items)

    @cfg.check_disable
    def add_outputs(self: Self, data: dict, name: str = '', is_save=True):
        items = Items(item_dict=data, name=name)
        index = len(self.outputs)
        self.seq.append(Sequence(SequenceType.OUTPUTS, index))
        if is_save:
            output_dir = os.sep.join([self.outputs_dir, str(index)])
            items.save_to_file(output_dir)
        self.outputs.append(items)

    @cfg.check_disable
    def add_extractor(self: Self, extractor: Self):
        index = len(self.extractors)
        self.seq.append(Sequence(SequenceType.EXTRACTORS, index, extractor.name))
        self.extractors.append(extractor)

    @cfg.check_disable
    def register_hook(self: Self, model: Module, name: str = '', is_warn_exist=False):
        if cfg.config['is_disable']:
            return
        if name == '':
            name = self.name
        if name not in self.nodes_by_name:
            node = NNNode(name=name, model=model)
            self.nodes_by_name[name] = node
        elif is_warn_exist:
            cfg.logger.warning(f'NNExtractor.register_hook: node name already registered: {name}')
        self.register_forward_hook(self, model, name)
        self.register_backward_hook(self, model, name)

    @cfg.check_disable
    def register_forward_hook(self: Self, model: Module, name: str = '', is_warn_exist=False):
        if name == '':
            name = self.name

        # already registered
        if self.current_forward_node is not None and self.current_forward_node.name == name:
            return

        if name in self.nodes_by_name:
            if is_warn_exist:
                cfg.logger.warning(f'NNExtractor.register_hook: node name already registered: {name}')
            node = self.nodes_by_name[name]
        else:
            node = NNNode(name=name, model=model)

        # try remove original forward node
        if self.current_forward_node is not None:
            self.current_forward_node.remove_forward_hook()
            self.current_forward_node = None

        # register
        node.register_forward_hook()
        self.current_forward_node = node

        if name not in self.nodes_by_name:
            self.nodes.append(node)
            self.nodes_by_name[name] = node

    @cfg.check_disable
    def forward_snapshot(self: Self):
        index = self.forward_snapshot_index
        self.seq.append(Sequence(SequenceType.FORWARD, index, self.current_forward_node.name))
        self.current_forward_node.forward_save_to_file(self.forward_dir, index)
        self.forward_snapshot_index += 1

    @cfg.check_disable
    def register_backward_hook(self: Self, model: Module, name: str = '', is_warn_exist=False):
        if name == '':
            name = self.name

        # already registered
        if self.current_backward_node is not None and self.current_backward_node.name == name:
            return

        if name in self.nodes_by_name:
            if is_warn_exist:
                cfg.logger.warning(f'NNExtractor.register_hook: node name already registered: {name}')
            node = self.nodes_by_name[name]
        else:
            node = NNNode(name=name, model=model)

        # try remove original backward node
        if self.current_backward_node is not None:
            self.current_backward_node.remove_backward_hook()
            self.current_backward_node = None

        # register
        node.register_backward_hook()
        self.current_backward_node = node

        if name not in self.nodes_by_name:
            self.nodes.append(node)
            self.nodes_by_name[name] = node

    @cfg.check_disable
    def backward_snapshot(self: Self):
        index = self.backward_snapshot_index
        self.seq.append(Sequence(SequenceType.BACKWARD, index, self.current_backward_node.name))
        self.current_backward_node.forward_save_to_file(self.backward_dir, index)
        self.backward_snapshot_index += 1

    @cfg.check_disable
    def remove_hook(self: Self):
        self.remove_forward_hook()
        self.remove_backward_hook()

    @cfg.check_disable
    def remove_forward_hook(self: Self):
        if self.current_forward_node is not None:
            self.current_forward_node.remove_forward_hook()
            self.current_forward_node = None

    @cfg.check_disable
    def remove_backward_hook(self: Self):
        if self.current_backward_node is not None:
            self.current_backward_node.remove_backward_hook()
            self.current_backward_node = None

    @cfg.check_disable
    def save(self: Self, outputs_dir: str = ''):
        if outputs_dir == '':
            outputs_dir = self.root_dir

        self.save_meta(outputs_dir)

    def meta(self: Self) -> Optional[MetaNNExtractor]:
        meta_seq = [each.meta() for each in self.seq]
        meta_inputs = [each.meta() for each in self.inputs]
        meta_preprocesses = [each.meta() for each in self.preprocesses]
        meta_nodes = [each.meta() for each in self.nodes]
        meta_postprocesses = [each.meta() for each in self.postprocesses]
        meta_outputs = [each.meta() for each in self.outputs]
        meta_extractors = [each.meta() for each in self.extractors]

        return MetaNNExtractor(
            name=self.name,

            seq=meta_seq,
            inputs=meta_inputs,
            preprocesses=meta_preprocesses,
            nodes=meta_nodes,
            postprocesses=meta_postprocesses,
            outputs=meta_outputs,
            extractors=meta_extractors,
        )

    @cfg.check_disable
    def save_meta(self: Self, outputs_dir: str = ''):
        '''
        save to json for easy reading in js.
        '''
        the_meta = self.meta()
        out_filename = os.sep.join([outputs_dir, f'{self.name}.meta.json'])

        with open(out_filename, 'w') as f:
            json.dump(the_meta, f)
