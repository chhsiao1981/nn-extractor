# -*- coding: utf-8 -*-

import unittest
import logging  # noqa

from nn_extractor import item  # noqa
from nn_extractor.item import Item, ItemType
from nn_extractor import nnextractor_pb2


class TestItem(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_serialize(self):
        expected_bytes = b'\n\x01a\x10\x01"\x02\x08\x01'

        a = Item('a', ItemType.RAW, None)
        a_pb = a.serialize_pb()
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_serialize_2(self):
        expected_bytes = b'\n\x01a\x10\x02\x1a\x04none"\x02\x08\x01'

        a = Item('a', ItemType.OTHER, None, other_type='none')
        a_pb = a.serialize_pb()
        a_bytes = a_pb.SerializeToString()

        print(f'a_bytes: {a_bytes}')

        assert a_bytes == expected_bytes

    def test_deserialize(self):
        a_bytes = b'\n\x01a\x10\x01"\x02\x08\x01'

        expected_a = Item('a', ItemType.RAW, None)

        a_pb = nnextractor_pb2.Item.FromString(a_bytes)
        a = Item.deserialize_pb(a_pb)

        assert a == expected_a
