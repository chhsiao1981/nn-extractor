# -*- coding: utf-8 -*-

import unittest
import logging

import numpy as np

from nn_extractor import constants
from nn_extractor import ndarray
from nn_extractor import nnextractor_pb2


class TestNdarray(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize_ndarray_bool(self):
        expected_a_bytes = b"\n\x01B:$\n\x01B:\t\n\x01b\x12\x04\x01\x01\x01\x01:\t\n\x01b\x12\x04\x01\x01\x01\x00:\t\n\x01b\x12\x04\x01\x01\x00\x00:$\n\x01B:\t\n\x01b\x12\x04\x01\x00\x00\x00:\t\n\x01b\x12\x04\x01\x00\x01\x00:\t\n\x01b\x12\x04\x00\x00\x00\x00"

        a = np.array([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=np.bool)

        a_pb = ndarray.serialize_ndarray(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes:', a_bytes)

        assert a_bytes == expected_a_bytes

    def test_serialize_ndarray_int32(self):
        expected_a_bytes = b'\n\x01H:$\n\x01H:\t\n\x01h\x1a\x04\x01\x02\x03\x04:\t\n\x01h\x1a\x04\x05\x06\x07\x08:\t\n\x01h\x1a\x04\t\n\x0b\x0c:$\n\x01H:\t\n\x01h\x1a\x04\r\x0e\x0f\x10:\t\n\x01h\x1a\x04\x11\x12\x13\x14:\t\n\x01h\x1a\x04\x15\x16\x17\x18'

        a_int32 = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int32)

        a_pb = ndarray.serialize_ndarray(a_int32)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes:', a_bytes)

        assert a_bytes == expected_a_bytes

    def test_serialize_ndarray_int64(self):
        expected_a_bytes = b'\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_pb = ndarray.serialize_ndarray(a)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes:', a_bytes)

        assert a_bytes == expected_a_bytes

    def test_serialize_ndarray_float32(self):
        expected_a_bytes = b'\n\x01E:H\n\x01E:\x15\n\x01e*\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@:\x15\n\x01e*\x10\x00\x00\xa0@\x00\x00\xc0@\x00\x00\xe0@\x00\x00\x00A:\x15\n\x01e*\x10\x00\x00\x10A\x00\x00 A\x00\x000A\x00\x00@A:H\n\x01E:\x15\n\x01e*\x10\x00\x00PA\x00\x00`A\x00\x00pA\x00\x00\x80A:\x15\n\x01e*\x10\x00\x00\x88A\x00\x00\x90A\x00\x00\x98A\x00\x00\xa0A:\x15\n\x01e*\x10\x00\x00\xa8A\x00\x00\xb0A\x00\x00\xb8A\x00\x00\xc0A'

        a_float32 = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float32)

        a_pb = ndarray.serialize_ndarray(a_float32)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes:', a_bytes)

        assert a_bytes == expected_a_bytes

    def test_serialize_ndarray_float64(self):
        expected_a_bytes = b'\n\x01F:x\n\x01F:%\n\x01f2 \x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@:%\n\x01f2 \x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @:%\n\x01f2 \x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00&@\x00\x00\x00\x00\x00\x00(@:x\n\x01F:%\n\x01f2 \x00\x00\x00\x00\x00\x00*@\x00\x00\x00\x00\x00\x00,@\x00\x00\x00\x00\x00\x00.@\x00\x00\x00\x00\x00\x000@:%\n\x01f2 \x00\x00\x00\x00\x00\x001@\x00\x00\x00\x00\x00\x002@\x00\x00\x00\x00\x00\x003@\x00\x00\x00\x00\x00\x004@:%\n\x01f2 \x00\x00\x00\x00\x00\x005@\x00\x00\x00\x00\x00\x006@\x00\x00\x00\x00\x00\x007@\x00\x00\x00\x00\x00\x008@'

        a_float64 = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float64)

        a_pb = ndarray.serialize_ndarray(a_float64)
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes:', a_bytes)

        assert a_bytes == expected_a_bytes

    def test_deserialize_ndarray_bool(self):
        a_bytes = b"\n\x01B:$\n\x01B:\t\n\x01b\x12\x04\x01\x01\x01\x01:\t\n\x01b\x12\x04\x01\x01\x01\x00:\t\n\x01b\x12\x04\x01\x01\x00\x00:$\n\x01B:\t\n\x01b\x12\x04\x01\x00\x00\x00:\t\n\x01b\x12\x04\x01\x00\x01\x00:\t\n\x01b\x12\x04\x00\x00\x00\x00"

        a_pb = nnextractor_pb2.NDArray.FromString(a_bytes)

        expected_a = np.array([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=np.bool)

        a = ndarray.deserialize_ndarray(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_ndarray_int32(self):
        a_bytes = b'\n\x01H:$\n\x01H:\t\n\x01h\x1a\x04\x01\x02\x03\x04:\t\n\x01h\x1a\x04\x05\x06\x07\x08:\t\n\x01h\x1a\x04\t\n\x0b\x0c:$\n\x01H:\t\n\x01h\x1a\x04\r\x0e\x0f\x10:\t\n\x01h\x1a\x04\x11\x12\x13\x14:\t\n\x01h\x1a\x04\x15\x16\x17\x18'

        a_pb = nnextractor_pb2.NDArray.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int32)

        a = ndarray.deserialize_ndarray(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_ndarray_int64(self):
        a_bytes = b'\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a_pb = nnextractor_pb2.NDArray.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a = ndarray.deserialize_ndarray(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_ndarray_float32(self):
        a_bytes = b'\n\x01E:H\n\x01E:\x15\n\x01e*\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@:\x15\n\x01e*\x10\x00\x00\xa0@\x00\x00\xc0@\x00\x00\xe0@\x00\x00\x00A:\x15\n\x01e*\x10\x00\x00\x10A\x00\x00 A\x00\x000A\x00\x00@A:H\n\x01E:\x15\n\x01e*\x10\x00\x00PA\x00\x00`A\x00\x00pA\x00\x00\x80A:\x15\n\x01e*\x10\x00\x00\x88A\x00\x00\x90A\x00\x00\x98A\x00\x00\xa0A:\x15\n\x01e*\x10\x00\x00\xa8A\x00\x00\xb0A\x00\x00\xb8A\x00\x00\xc0A'

        a_pb = nnextractor_pb2.NDArray.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float32)

        a = ndarray.deserialize_ndarray(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_ndarray_float64(self):
        a_bytes = b'\n\x01F:x\n\x01F:%\n\x01f2 \x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@:%\n\x01f2 \x00\x00\x00\x00\x00\x00\x14@\x00\x00\x00\x00\x00\x00\x18@\x00\x00\x00\x00\x00\x00\x1c@\x00\x00\x00\x00\x00\x00 @:%\n\x01f2 \x00\x00\x00\x00\x00\x00"@\x00\x00\x00\x00\x00\x00$@\x00\x00\x00\x00\x00\x00&@\x00\x00\x00\x00\x00\x00(@:x\n\x01F:%\n\x01f2 \x00\x00\x00\x00\x00\x00*@\x00\x00\x00\x00\x00\x00,@\x00\x00\x00\x00\x00\x00.@\x00\x00\x00\x00\x00\x000@:%\n\x01f2 \x00\x00\x00\x00\x00\x001@\x00\x00\x00\x00\x00\x002@\x00\x00\x00\x00\x00\x003@\x00\x00\x00\x00\x00\x004@:%\n\x01f2 \x00\x00\x00\x00\x00\x005@\x00\x00\x00\x00\x00\x006@\x00\x00\x00\x00\x00\x007@\x00\x00\x00\x00\x00\x008@'

        a_pb = nnextractor_pb2.NDArray.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float64)

        a = ndarray.deserialize_ndarray(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_meta_ndarray_bool(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_BOOL,
        }

        a = np.array([
            [[True, True, True, True], [True, True, True, False], [True, True, False, False]],
            [[True, False, False, False], [True, False, True, False], [False, False, False, False]],
        ], dtype=np.bool)

        meta = ndarray.meta_ndarray(a)

        assert meta == expected_meta

    def test_meta_ndarray_int32(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_INT32,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int32)

        meta = ndarray.meta_ndarray(a)

        assert meta == expected_meta

    def test_meta_ndarray_int64(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_INT64,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        meta = ndarray.meta_ndarray(a)

        assert meta == expected_meta

    def test_meta_ndarray_float32(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_FLOAT32,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float32)

        meta = ndarray.meta_ndarray(a)

        assert meta == expected_meta

    def test_meta_ndarray_float64(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_FLOAT64,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.float64)

        meta = ndarray.meta_ndarray(a)

        assert meta == expected_meta
