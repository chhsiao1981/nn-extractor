# -*- coding: utf-8 -*-

import unittest
import logging

import numpy as np

from nn_extractor import constants
from nn_extractor import record
from nn_extractor import nnextractor_pb2


class TestRecord(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize_record_ndarray(self):
        expected_a_bytes = b'\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_pb = record.serialize_record(a)

        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_record_list(self):
        expected_a_bytes = b'\n\x01l\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_list = [a, a, a, a, a]

        a_pb = record.serialize_record(a_list)

        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_record_tuple(self):
        expected_a_bytes = b'\n\x01t\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_tuple = (a, a, a, a, a)

        a_pb = record.serialize_record(a_tuple)

        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_serialize_record_dict(self):
        expected_a_bytes = b'\n\x01d"Y\n\x01c\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01b\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01a\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01e\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01d\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_dict = {'a': a, 'b': a, 'c': a, 'd': a, 'e': a}

        a_pb = record.serialize_record(a_dict)

        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_a_bytes

    def test_deserialize_record_ndarray(self):
        a_bytes = b'\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a_pb = nnextractor_pb2.Record.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a = record.deserialize_record(a_pb)

        assert (a == expected_a).all() and np.isdtype(a.dtype, expected_a.dtype)

    def test_deserialize_record_list(self):
        a_bytes = b'\n\x01l\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a_pb = nnextractor_pb2.Record.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        expected_a_list = [expected_a, expected_a, expected_a, expected_a, expected_a]

        a_list = record.deserialize_record(a_pb)

        assert type(a_list) == type(expected_a_list)
        for idx, each in enumerate(a_list):
            expected_each = expected_a_list[idx]
            assert (each == expected_each).all() and np.isdtype(each.dtype, expected_each.dtype)

    def test_deserialize_record_tuple(self):
        a_bytes = b'\n\x01t\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18\x1aT\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a_pb = nnextractor_pb2.Record.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        expected_a_tuple = tuple([expected_a, expected_a, expected_a, expected_a, expected_a])

        a_tuple = record.deserialize_record(a_pb)

        assert type(a_tuple) == type(expected_a_tuple)
        for idx, each in enumerate(a_tuple):
            expected_each = expected_a_tuple[idx]
            assert (each == expected_each).all() and np.isdtype(each.dtype, expected_each.dtype)

    def test_deserialize_record_dict(self):
        a_bytes = b'\n\x01d"Y\n\x01c\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01b\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01a\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01e\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18"Y\n\x01d\x12T\n\x01a\x12O\n\x01I:$\n\x01I:\t\n\x01i"\x04\x01\x02\x03\x04:\t\n\x01i"\x04\x05\x06\x07\x08:\t\n\x01i"\x04\t\n\x0b\x0c:$\n\x01I:\t\n\x01i"\x04\r\x0e\x0f\x10:\t\n\x01i"\x04\x11\x12\x13\x14:\t\n\x01i"\x04\x15\x16\x17\x18'

        a_pb = nnextractor_pb2.Record.FromString(a_bytes)

        expected_a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        expected_a_dict = {'a': expected_a, 'b': expected_a, 'c': expected_a, 'd': expected_a, 'e': expected_a}

        a_dict = record.deserialize_record(a_pb)

        assert type(a_dict) == type(expected_a_dict)
        for key, each in a_dict.items():
            expected_each = expected_a_dict[key]
            assert (each == expected_each).all() and np.isdtype(each.dtype, expected_each.dtype)

    def test_meta_record_ndarray(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_INT64,
        }

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        meta = record.meta_record(a)

        assert meta == expected_meta

    def test_meta_record_list(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_INT64,
        }

        expected_meta_list = [expected_meta, expected_meta, expected_meta, expected_meta, expected_meta]

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_list = [a, a, a, a, a]

        meta = record.meta_record(a_list)

        assert meta == expected_meta_list

    def test_meta_record_tuple(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_INT64,
        }

        expected_meta_tuple = tuple([expected_meta, expected_meta, expected_meta, expected_meta, expected_meta])

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_tuple = tuple([a, a, a, a, a])

        meta = record.meta_record(a_tuple)

        assert meta == expected_meta_tuple

    def test_meta_record_dict(self):
        expected_meta = {
            'shape': (2, 3, 4),
            'type': constants.META_INT64,
        }

        expected_meta_dict = {'a': expected_meta, 'b': expected_meta, 'c': expected_meta, 'd': expected_meta, 'e': expected_meta}

        a = np.array([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ], dtype=np.int64)

        a_dict = {'a': a, 'b': a, 'c': a, 'd': a, 'e': a}

        meta = record.meta_record(a_dict)

        assert meta == expected_meta_dict
