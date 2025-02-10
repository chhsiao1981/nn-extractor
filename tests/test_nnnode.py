# -*- coding: utf-8 -*-

from typing import Self

import unittest
import logging  # noqa


from nn_extractor.ops.op_item import OpItem
from nn_extractor.ops.op_type import OpType
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from nn_extractor import nnnode  # noqa

from nn_extractor.nnrecord import NNRecord
from nn_extractor.nnparameter import NNParameter
from nn_extractor.nnnode import NNNode

from nn_extractor import nnextractor_pb2


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

        self.bias = Parameter(torch.Tensor([1, 2, 3]))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print(f'after conv1: x: ({type(x)}/{x.shape})')
        ret = F.relu(self.conv2(x))
        print(f'after conv2: ret: ({type(ret)}/{ret.shape})')
        return ret


class TestNNNode(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize_null_inputs(self):
        expected_bytes = b'\n\x01a'
        a = NNNode(name='a')

        a_pb = a.forward_serialize_pb()
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_deserialize_null_inputs(self):
        a_bytes = b'\n\x01a'
        expected_a = NNNode(name='a')

        a_pb = nnextractor_pb2.NNNode.FromString(a_bytes)

        a = NNNode.deserialize_pb(a_pb)

        print(f'a: {a} expected_a: {expected_a}')

        assert a == expected_a

    def test_register_forward(self):
        model = Model()
        named_children = list(model.named_children())

        named_children_params_map = {name: self._params_map(
            child) for name, child in named_children}

        print(f'children: names: {named_children_params_map.keys()}')

        a = NNNode(name='a', model=model)
        a.register_forward_hook()

        x = torch.ones(1, 40, 40)
        x_np = x.detach().to('cpu').numpy()
        print(f'x_np: ({x_np}/{x_np.shape})')
        model(x)

        params0 = NNParameter(name='bias', parameter=np.array([1, 2, 3]))

        weight0 = named_children_params_map[named_children[0][0]]['weight']
        bias0 = named_children_params_map[named_children[0][0]]['bias']
        print(f'weight0: {weight0.shape} bias0: {bias0.shape}')
        children0_params0 = NNParameter(name='weight', parameter=weight0)
        children0_params1 = NNParameter(name='bias', parameter=bias0)

        x_batch = x[None, :, :, :]
        output0 = F.conv2d(x_batch, torch.Tensor(weight0), torch.Tensor(bias0))[0]
        output0_np = output0.numpy()
        output0_op_item = OpItem(
            name='a.conv1',
            op_type=OpType.SPACING,
            tensor=output0_np,
            op_params=None,
        )
        output0_op_item.set_data_id('./0/a.conv1/activation')
        output0_record = NNRecord(
            value=output0_op_item,
            data_id='./0/a.conv1/activation'
        )
        print(f'output0: {output0_np.shape} children: {a.children[0].activation.value.tensor.shape}')  # noqa

        input1 = F.relu(output0)
        input1_np = input1.numpy()
        input1_op_item = OpItem(
            name='a.conv2',
            op_type=OpType.SPACING,
            tensor=input1_np,
            op_params=None,
        )
        input1_op_item.set_data_id('./1/a.conv2/args/0')
        input1_record = NNRecord(
            value=input1_op_item,
            data_id='./1/a.conv2/args/0',
        )

        input1_batch = input1[None, :, :, :]
        weight1 = named_children_params_map[named_children[1][0]]['weight']
        bias1 = named_children_params_map[named_children[1][0]]['bias']
        children1_params0 = NNParameter(name='weight', parameter=weight1)
        children1_params1 = NNParameter(name='bias', parameter=bias1)
        output1 = F.conv2d(input1_batch, torch.Tensor(weight1), torch.Tensor(bias1))[0]
        output1_np = output1.numpy()
        output1_op_item = OpItem(
            name='a.conv2',
            op_type=OpType.SPACING,
            tensor=output1_np,
            op_params=None,
        )
        output1_op_item.set_data_id('./1/a.conv2/activation')
        output1_record = NNRecord(
            value=output1_op_item,
            data_id='./1/a.conv2/activation',
        )
        print(f'output1: {output1.shape} children: {a.children[1].activation.value.tensor.shape}')  # noqa

        output = F.relu(output1)
        output_np = output.numpy()
        output_op_item = OpItem(
            name='a',
            op_type=OpType.SPACING,
            tensor=output_np,
            op_params=None,
        )
        output_op_item.set_data_id('./a/activation')
        output_record = NNRecord(
            value=output_op_item,
            data_id='./a/activation',
        )

        x_np = x.detach().to('cpu').numpy()
        x_op_item = OpItem(name='a', op_type=OpType.SPACING, tensor=x_np, op_params=None)
        x_op_item.set_data_id('./a/args/0')
        x_record = NNRecord(
            value=x_op_item,
            data_id='./a/args/0'
        )

        children0_x_op_item = OpItem(
            name='a.conv1',
            op_type=OpType.SPACING,
            tensor=x_np,
            op_params=None,
        )
        children0_x_op_item.set_data_id('./0/a.conv1/args/0')
        children0_x_record = NNRecord(
            value=children0_x_op_item,
            data_id='./0/a.conv1/args/0'
        )

        assert a.inputs == [x_record]
        assert a.children[0].inputs == [children0_x_record]
        assert a.children[0].params[0] == children0_params0
        assert a.children[0].params[1] == children0_params1
        assert a.children[0].activation.name == output0_record.name
        assert a.children[0].activation.value.name == output0_record.value.name
        assert a.children[0].activation.value.op_type == output0_record.value.op_type
        assert (a.children[0].activation.value.tensor == output0_record.value.tensor).all()
        assert a.children[0].activation.value.op_params == output0_record.value.op_params
        assert a.children[0].activation.the_type == output0_record.the_type
        assert a.children[0].activation.data_id == output0_record.data_id

        assert a.children[1].inputs == [input1_record]
        assert a.children[1].params[0] == children1_params0
        assert a.children[1].params[1] == children1_params1
        assert a.children[1].activation.name == output1_record.name
        assert a.children[1].activation.value.name == output1_record.value.name
        assert a.children[1].activation.value.op_type == output1_record.value.op_type
        assert (a.children[1].activation.value.tensor == output1_record.value.tensor).all()
        assert a.children[1].activation.value.op_params == output1_record.value.op_params
        assert a.children[1].activation.the_type == output1_record.the_type
        assert a.children[1].activation.data_id == output1_record.data_id

        assert a.params[0] == params0

        assert a.activation.name == output_record.name
        assert a.activation.value.name == output_record.value.name
        assert a.activation.value.op_type == output_record.value.op_type
        assert (a.activation.value.tensor == output_record.value.tensor).all()
        assert a.activation.value.op_params == output_record.value.op_params
        assert a.activation.the_type == output_record.the_type
        assert a.activation.data_id == output_record.data_id

    def _params_map(self: Self, model: Module) -> dict[str, np.ndarray]:
        named_params = list(model.named_parameters())
        return {name: params.detach().to('cpu').numpy() for name, params in named_params}
