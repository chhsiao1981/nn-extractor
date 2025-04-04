# -*- coding: utf-8 -*-

import unittest
import logging  # noqa

import numpy as np
import torch

from nn_extractor import constants
from nn_extractor import nntensor
from nn_extractor import nnextractor_pb2


class TestNNTensor(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize_nntensor_bool(self):
        expected_a_bytes = b"\n\x03\x04\x06\x08\x12\x98\x02\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x06\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x04\x00\x04\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x13\x05\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00"  # noqa

        a = np.array([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=np.bool)

        a_pb = nntensor.serialize_nntensor_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_nntensor_int32(self):
        expected_a_bytes = b'\n\x03\x04\x06\x08\x12\xf8\x02\xff\xff\xff\xffx\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x02\x10\x00\x00\x00\x1c\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x08\x00\x07\x00\x08\x00\x00\x00\x00\x00\x00\x01 \x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\t\x00\x00\x00\n\x00\x00\x00\x0b\x00\x00\x00\x0c\x00\x00\x00\r\x00\x00\x00\x0e\x00\x00\x00\x0f\x00\x00\x00\x10\x00\x00\x00\x11\x00\x00\x00\x12\x00\x00\x00\x13\x00\x00\x00\x14\x00\x00\x00\x15\x00\x00\x00\x16\x00\x00\x00\x17\x00\x00\x00\x18\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a_int32 = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int32)

        a_pb = nntensor.serialize_nntensor_pb(a_int32)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_nntensor_int64(self):
        expected_a_bytes = b'\n\x03\x04\x06\x08\x12\xd8\x03\xff\xff\xff\xffx\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x02\x10\x00\x00\x00\x1c\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x08\x00\x07\x00\x08\x00\x00\x00\x00\x00\x00\x01@\x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x00\x00\n\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_pb = nntensor.serialize_nntensor_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_nntensor_float32(self):
        expected_a_bytes = b'\n\x03\x04\x06\x08\x12\xf0\x02\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x03\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x08\x00\x06\x00\x06\x00\x00\x00\x00\x00\x01\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x00\x00\xa0@\x00\x00\xc0@\x00\x00\xe0@\x00\x00\x00A\x00\x00\x10A\x00\x00 A\x00\x000A\x00\x00@A\x00\x00PA\x00\x00`A\x00\x00pA\x00\x00\x80A\x00\x00\x88A\x00\x00\x90A\x00\x00\x98A\x00\x00\xa0A\x00\x00\xa8A\x00\x00\xb0A\x00\x00\xb8A\x00\x00\xc0A\xff\xff\xff\xff\x00\x00\x00\x00'   # noqa

        a_float32 = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float32)

        a_pb = nntensor.serialize_nntensor_pb(a_float32)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_nntensor_float64(self):
        expected_a_bytes = b'\n\x03\x04\x06\x08\x12\xd0\x03\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x03\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x08\x00\x06\x00\x06\x00\x00\x00\x00\x00\x02\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00&@\x00\x00\x00\x00\x00\x00(@\x00\x00\x00\x00\x00\x00*@\x00\x00\x00\x00\x00\x00,@\x00\x00\x00\x00\x00\x00.@\x00\x00\x00\x00\x00\x000@\x00\x00\x00\x00\x00\x001@\x00\x00\x00\x00\x00\x002@\x00\x00\x00\x00\x00\x003@\x00\x00\x00\x00\x00\x004@\x00\x00\x00\x00\x00\x005@\x00\x00\x00\x00\x00\x006@\x00\x00\x00\x00\x00\x007@\x00\x00\x00\x00\x00\x008@\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a_float64 = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float64)

        a_pb = nntensor.serialize_nntensor_pb(a_float64)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_deserialize_nntensor_bool(self):
        a_bytes = b"\n\x03\x04\x06\x08\x12\x98\x02\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x06\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x04\x00\x04\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x13\x05\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00"  # noqa

        a_pb = nnextractor_pb2.NNTensor.FromString(a_bytes)

        expected_a = np.array([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=np.bool)

        a = nntensor.deserialize_nntensor_pb(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_nntensor_int32(self):
        a_bytes = b'\n\x03\x04\x06\x08\x12\xf8\x02\xff\xff\xff\xffx\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x02\x10\x00\x00\x00\x1c\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x08\x00\x07\x00\x08\x00\x00\x00\x00\x00\x00\x01 \x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\t\x00\x00\x00\n\x00\x00\x00\x0b\x00\x00\x00\x0c\x00\x00\x00\r\x00\x00\x00\x0e\x00\x00\x00\x0f\x00\x00\x00\x10\x00\x00\x00\x11\x00\x00\x00\x12\x00\x00\x00\x13\x00\x00\x00\x14\x00\x00\x00\x15\x00\x00\x00\x16\x00\x00\x00\x17\x00\x00\x00\x18\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a_pb = nnextractor_pb2.NNTensor.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int32)

        a = nntensor.deserialize_nntensor_pb(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_nntensor_int64(self):
        a_bytes = b'\n\x03\x04\x06\x08\x12\xd8\x03\xff\xff\xff\xffx\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x02\x10\x00\x00\x00\x1c\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\x00\x0c\x00\x08\x00\x07\x00\x08\x00\x00\x00\x00\x00\x00\x01@\x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x00\x00\n\x00\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x00\x00\x00\x00\r\x00\x00\x00\x00\x00\x00\x00\x0e\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x15\x00\x00\x00\x00\x00\x00\x00\x16\x00\x00\x00\x00\x00\x00\x00\x17\x00\x00\x00\x00\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a_pb = nnextractor_pb2.NNTensor.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a = nntensor.deserialize_nntensor_pb(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_nntensor_float32(self):
        a_bytes = b'\n\x03\x04\x06\x08\x12\xf0\x02\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x03\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x08\x00\x06\x00\x06\x00\x00\x00\x00\x00\x01\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x00\x00\xa0@\x00\x00\xc0@\x00\x00\xe0@\x00\x00\x00A\x00\x00\x10A\x00\x00 A\x00\x000A\x00\x00@A\x00\x00PA\x00\x00`A\x00\x00pA\x00\x00\x80A\x00\x00\x88A\x00\x00\x90A\x00\x00\x98A\x00\x00\xa0A\x00\x00\xa8A\x00\x00\xb0A\x00\x00\xb8A\x00\x00\xc0A\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a_pb = nnextractor_pb2.NNTensor.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float32)

        a = nntensor.deserialize_nntensor_pb(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_nntensor_float64(self):
        a_bytes = b'\n\x03\x04\x06\x08\x12\xd0\x03\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x03\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x06\x00\x08\x00\x06\x00\x06\x00\x00\x00\x00\x00\x02\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @\x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00&@\x00\x00\x00\x00\x00\x00(@\x00\x00\x00\x00\x00\x00*@\x00\x00\x00\x00\x00\x00,@\x00\x00\x00\x00\x00\x00.@\x00\x00\x00\x00\x00\x000@\x00\x00\x00\x00\x00\x001@\x00\x00\x00\x00\x00\x002@\x00\x00\x00\x00\x00\x003@\x00\x00\x00\x00\x00\x004@\x00\x00\x00\x00\x00\x005@\x00\x00\x00\x00\x00\x006@\x00\x00\x00\x00\x00\x007@\x00\x00\x00\x00\x00\x008@\xff\xff\xff\xff\x00\x00\x00\x00'  # noqa

        a_pb = nnextractor_pb2.NNTensor.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float64)

        a = nntensor.deserialize_nntensor_pb(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_meta_nntensor_bool(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'the_type': constants.META_BOOL,
        }

        a = np.array([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=np.bool)

        meta = nntensor.meta_nntensor(a)

        assert meta == expected_meta

    def test_meta_nntensor_int32(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'the_type': constants.META_INT32,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int32)

        meta = nntensor.meta_nntensor(a)

        assert meta == expected_meta

    def test_meta_nntensor_int64(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'the_type': constants.META_INT64,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        meta = nntensor.meta_nntensor(a)

        assert meta == expected_meta

    def test_meta_nntensor_float32(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'the_type': constants.META_FLOAT32,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float32)

        meta = nntensor.meta_nntensor(a)

        assert meta == expected_meta

    def test_meta_nntensor_float64(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'the_type': constants.META_FLOAT64,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float64)

        meta = nntensor.meta_nntensor(a)

        assert meta == expected_meta

    def test_serialize_nntensor_bool_torch(self):
        expected_a_bytes = b"\n\x03\x04\x06\x08\x12\x98\x02\xff\xff\xff\xffp\x00\x00\x00\x10\x00\x00\x00\x00\x00\n\x00\x0c\x00\x06\x00\x05\x00\x08\x00\n\x00\x00\x00\x00\x01\x04\x00\x0c\x00\x00\x00\x08\x00\x08\x00\x00\x00\x04\x00\x08\x00\x00\x00\x04\x00\x00\x00\x01\x00\x00\x00\x14\x00\x00\x00\x10\x00\x14\x00\x08\x00\x06\x00\x07\x00\x0c\x00\x00\x00\x10\x00\x10\x00\x00\x00\x00\x00\x01\x06\x10\x00\x00\x00\x18\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x04\x00\x04\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff\x88\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x16\x00\x06\x00\x05\x00\x08\x00\x0c\x00\x0c\x00\x00\x00\x00\x03\x04\x00\x18\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\n\x00\x18\x00\x0c\x00\x04\x00\x08\x00\n\x00\x00\x00<\x00\x00\x00\x10\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x18\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x13\x05\x00\x00\x00\x00\x00\xff\xff\xff\xff\x00\x00\x00\x00"  # noqa

        a = torch.tensor([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=torch.bool)

        a_pb = nntensor.serialize_nntensor_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes
