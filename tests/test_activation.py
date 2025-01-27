# -*- coding: utf-8 -*-

import unittest
import logging

import numpy as np

from nn_extractor import nnextractor_pb2
from nn_extractor import activation
from nn_extractor.parameter import Parameter
from nn_extractor.activation import Activation


class TestActivation(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize_activation(self):
        expected_a_bytes = b'\n\x01a\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1a^\n\x06params\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'
        the_input = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        the_record = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        params = ('params', the_record)

        the_output = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a = {
            'name': 'a',
            'input': the_input,
            'params': [params],
            'output': the_output,
        }

        a_pb = activation.serialize_activation(a)

        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_deserialize_activation(self):
        the_input = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        the_record = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        params = Parameter(name='params', record=the_record)

        the_output = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        expected_a = Activation(name='a', input=the_input, params=[params], output=the_output)

        a_bytes = b'\n\x01a\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1a^\n\x06params\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'
        a_pb = nnextractor_pb2.Activation.FromString(a_bytes)

        a = activation.deserialize_activation(a_pb)

        assert activation.is_same_activation(a, expected_a)
