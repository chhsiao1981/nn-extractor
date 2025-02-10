# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, Optional

import json
from torch.nn import Module

import os

from . import cfg
from . import constants
from .items import Items

from .nnnode import NNNode

from .taskflow import Taskflow, TaskflowType

from .types import MetaNNExtractor
from . import utils


@dataclass
class NNExtractor(object):
    '''
    NNExtractor

    reqiring the nodes to be unique.
    '''
    name: str

    taskflow: list[Taskflow]
    inputs: list[Items]
    preprocesses: list[Items]
    nodes: list[NNNode]
    postprocesses: list[Items]
    outputs: list[Items]

    nodes_by_name: dict[str, NNNode]

    extractors: list[Self]

    parent: Optional[Self]

    '''
    the root-dir of this extractor: same as the name.
    '''
    root_dir: str

    '''
    the inputs-dir of this extractor: [root_dir]/[DIR_INPUTS (inputs)]
    '''
    input_dir: str

    '''
    the preprocess-dir of this extractor: [root_dir]/[DIR_PREPROCESSES (preprocesses)]
    '''
    preprocess_dir: str

    '''
    the forward-dir of this extractor: [root_dir]/[DIR_FORWARDS (forwards)]
    '''
    forward_dir: str

    '''
    the backward-dir of this extractor: [root_dir]/[DIR_BACKWARDS (backwards)]
    '''
    backward_dir: str

    '''
    the postprocess-dir of this extractor: [root_dir]/[DIR_POSTPROCESSES (postprocesses)]
    '''
    postprocess_dir: str

    '''
    the outputs-dir of this extractor: [root_dir]/[DIR_OUTPUTS (outputs)]
    '''
    output_dir: str

    current_forward_node: Optional[NNNode] = None
    current_backward_node: Optional[NNNode] = None

    forward_snapshot_index: int = 0
    backward_snapshot_index: int = 0

    def __init__(self, name: str, parent: Optional[Self] = None):
        self.name = name
        self.taskflow = []
        self.inputs = []
        self.preprocesses = []
        self.nodes = []
        self.postprocesses = []
        self.outputs = []

        self.nodes_by_name = {}
        self.extractors = []

        self.parent = parent

        self.root_dir = name
        self.input_dir = os.sep.join([self.root_dir, f'{TaskflowType.INPUT}'])
        self.preprocess_dir = os.sep.join([self.root_dir, f'{TaskflowType.PREPROCESS}'])
        self.forward_dir = os.sep.join([self.root_dir, f'{TaskflowType.FORWARD}'])
        self.backward_dir = os.sep.join([self.root_dir, f'{TaskflowType.BACKWARD}'])
        self.postprocess_dir = os.sep.join([self.root_dir, f'{TaskflowType.POSTPROCESS}'])
        self.output_dir = os.sep.join([self.root_dir, f'{TaskflowType.OUTPUT}'])

    @cfg.check_disable
    def add_inputs(
        self: Self,
        data: dict,
        name: str = '',
        is_save=True,
    ):
        flow_id = len(self.inputs)
        items = Items(name=name, item_dict=data)

        if is_save:
            flow_dir = os.sep.join([self.input_dir, f'{flow_id}'])
            items.save_to_file(flow_dir)

        self.inputs.append(items)

        # add to seq
        self.taskflow.append(Taskflow(
            name=name,
            the_type=TaskflowType.INPUT,
            flow_id=flow_id,
            items=items,
        ))

    @cfg.check_disable
    def add_preprocess(
        self: Self,
        data: dict,
        name: str = '',
        is_save=True,
    ):
        flow_id = len(self.preprocesses)
        items = Items(name=name, item_dict=data)

        if is_save:
            flow_dir = os.sep.join([self.preprocess_dir, f'{flow_id}'])
            items.save_to_file(flow_dir)

        self.preprocesses.append(items)

        # add to seq
        self.taskflow.append(Taskflow(
            name=name,
            the_type=TaskflowType.PREPROCESS,
            flow_id=flow_id,
            items=items,
        ))

    @cfg.check_disable
    def add_postprocess(
        self: Self,
        data: dict,
        name: str = '',
        is_save=True,
    ):
        flow_id = len(self.postprocesses)
        items = Items(item_dict=data, name=name)

        if is_save:
            flow_dir = os.sep.join([self.postprocess_dir, f'{flow_id}'])
            items.save_to_file(flow_dir)

        self.postprocesses.append(items)

        # add to seq
        self.taskflow.append(Taskflow(
            name=name,
            the_type=TaskflowType.POSTPROCESS,
            flow_id=flow_id,
            items=items,
        ))

    @cfg.check_disable
    def add_outputs(
        self: Self,
        data: dict,
        name: str = '',
        is_save=True,
    ):
        flow_id = len(self.outputs)
        items = Items(item_dict=data, name=name)

        if is_save:
            flow_dir = os.sep.join([self.output_dir, f'{flow_id}'])
            items.save_to_file(flow_dir)

        self.outputs.append(items)

        # add to seq
        self.taskflow.append(Taskflow(
            name=name,
            the_type=TaskflowType.OUTPUT,
            flow_id=flow_id,
            items=items,
        ))

    @cfg.check_disable
    def add_extractor(self: Self, extractor: Self):
        flow_id = len(self.extractors)
        self.taskflow.append(Taskflow(
            name=extractor.name,
            the_type=TaskflowType.EXTRACTOR,
            flow_id=flow_id,
        ))
        self.extractors.append(extractor)
        extractor.parent = self

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
                cfg.logger.warning(
                    f'NNExtractor.register_hook: node name already registered: {name}')
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
        flow_id = self.forward_snapshot_index

        flow_dir = os.sep.join([self.forward_dir, f'{flow_id}'])
        self.current_forward_node.forward_save_to_file(flow_dir)

        self.forward_snapshot_index += 1

        # add to seq
        self.taskflow.append(Taskflow(
            the_type=TaskflowType.FORWARD,
            flow_id=flow_id,
            name=self.current_forward_node.name,
        ))

    @cfg.check_disable
    def register_backward_hook(self: Self, model: Module, name: str = '', is_warn_exist=False):
        if name == '':
            name = self.name

        # already registered
        if self.current_backward_node is not None and self.current_backward_node.name == name:
            return

        if name in self.nodes_by_name:
            if is_warn_exist:
                cfg.logger.warning(
                    f'NNExtractor.register_hook: node name already registered: {name}')
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
        flow_id = self.backward_snapshot_index

        flow_dir = os.sep.join([self.backward_dir, f'{flow_id}'])
        self.current_backward_node.backward_save_to_file(flow_dir)

        self.backward_snapshot_index += 1

        # add to seq
        self.taskflow.append(Taskflow(
            the_type=TaskflowType.BACKWARD,
            flow_id=flow_id,
            name=self.current_backward_node.name,
        ))

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
    def save(self: Self):
        self.save_meta()

    def meta(self: Self) -> Optional[MetaNNExtractor]:
        meta_taskflow = [each.meta() for each in self.taskflow]
        meta_inputs = [each.meta() for each in self.inputs]
        meta_preprocesses = [each.meta() for each in self.preprocesses]
        meta_nodes = [each.meta() for each in self.nodes]
        meta_postprocesses = [each.meta() for each in self.postprocesses]
        meta_outputs = [each.meta() for each in self.outputs]
        meta_extractors = [each.meta() for each in self.extractors]

        parent_name = '' if self.parent is None else self.parent.name

        return MetaNNExtractor(
            name=self.name,

            taskflow=meta_taskflow,
            inputs=meta_inputs,
            preprocesses=meta_preprocesses,
            nodes=meta_nodes,
            postprocesses=meta_postprocesses,
            outputs=meta_outputs,
            extractors=meta_extractors,

            parent_name=parent_name,
        )

    @cfg.check_disable
    def save_meta(self: Self):
        '''
        save to json for easy reading in js.
        '''
        the_meta = self.meta()
        out_filename = os.sep.join([
            cfg.config['output_dir'],
            self.root_dir,
            f'{self.name}.{constants.METADATA_FILENAME_PREFIX}.json',
        ])

        utils.ensure_dir(out_filename)
        with open(out_filename, 'w') as f:
            json.dump(the_meta, f, indent=2)
