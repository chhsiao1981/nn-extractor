# -*- coding: utf-8 -*-

import unittest

import numpy as np

import json
from nn_extractor import primitive
from nn_extractor.primitive import Primitive

from nn_extractor import nnextractor_pb2


class TestPrimitive(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize_none(self):
        expected_bytes = b'\x08\x01'

        a: Primitive = None

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        assert a_bytes == expected_bytes

    def test_serialize_bool(self):
        expected_bytes = b'\x08\x080\x02'
        a: Primitive = True

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_str(self):
        expected_bytes = b'\x08\x05\x1a\x03abc'
        a: Primitive = 'abc'

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_float(self):
        expected_bytes = b'\x08\x06!333333\xd3?'
        a: Primitive = 0.3

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_int(self):
        expected_bytes = b'\x08\x080\x06'
        a: Primitive = 3

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_bool(self):
        expected_bytes = b'\x08\x04\x10\x01'
        a: Primitive = np.bool(1)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_float64(self):
        expected_bytes = b'\x08\x06!333333\xd3?'
        a: Primitive = np.float64(0.3)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_float64_2(self):
        expected_bytes = b'\x08\x06!\x00\x00\x00\x00\x00\x00\xf0?'
        a: Primitive = np.float64(1)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_float32(self):
        expected_bytes = b'\x08\x07-\x9a\x99\x99>'
        a: Primitive = np.float32(0.3)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_float32_2(self):
        expected_bytes = b'\x08\x07-\x00\x00\x80?'
        a: Primitive = np.float32(1)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_int64(self):
        expected_bytes = b'\x08\x080\x06'
        a: Primitive = np.int64(3)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_int64_2(self):
        expected_bytes = b'\x08\x080\x02'
        a: Primitive = np.int64(1)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_int32(self):
        expected_bytes = b'\x08\t8\x06'
        a: Primitive = np.int32(3)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_np_int32_2(self):
        expected_bytes = b'\x08\t8\x02'
        a: Primitive = np.int32(1)

        a_pb = primitive.serialize_primitive_pb(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_json_dump(self):
        a: Primitive = 'a'

        print(f'test_json_dump: to dump a: {a}')
        a_str = json.dumps(a)

        assert a_str == '"a"'

    def test_deserialize_none(self):
        a_bytes = b'\x08\x01'

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a is None

    def test_deserialize_bool(self):
        a_bytes = b'\x08\x080\x02'
        expected_a: Primitive = True

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a

    def test_deserialize_str(self):
        a_bytes = b'\x08\x05\x1a\x03abc'
        expected_a: Primitive = 'abc'

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a

    def test_deserialize_float(self):
        a_bytes = b'\x08\x06!333333\xd3?'
        expected_a: Primitive = 0.3

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, float)

    def test_deserialize_int(self):
        a_bytes = b'\x08\x080\x06'
        expected_a: Primitive = 3

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, int)

    def test_deserialize_int_2(self):
        a_bytes = b'\x08\x080\x02'
        expected_a: Primitive = 1

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, int)

    def test_deserialize_np_float64_2(self):
        a_bytes = b'\x08\x06!\x00\x00\x00\x00\x00\x00\xf0?'
        expected_a: Primitive = 1.0

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, float)

    def test_deserialize_np_float32(self):
        a_bytes = b'\x08\x07-\x9a\x99\x99>'
        expected_a: Primitive = np.float32(0.3)

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, np.float32)

    def test_deserialize_np_float32_2(self):
        a_bytes = b'\x08\x07-\x00\x00\x80?'
        expected_a: Primitive = np.float32(1)

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, np.float32)

    def test_deserialize_np_int32(self):
        a_bytes = b'\x08\t8\x06'
        expected_a: Primitive = np.int32(3)

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, np.int32)

    def test_deserialize_np_int32_2(self):
        a_bytes = b'\x08\t8\x02'
        expected_a: Primitive = np.int32(1)

        a_pb = nnextractor_pb2.Primitive.FromString(a_bytes)
        a = primitive.deserialize_primitive_pb(a_pb)

        assert a == expected_a and isinstance(a, np.int32)

    def test_eq_np(self):
        a: Primitive = np.int32(1)
        b: Primitive = 3

        assert not primitive.eq_primitive(a, b)
