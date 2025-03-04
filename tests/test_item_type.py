# -*- coding: utf-8 -*-

import unittest
import logging  # noqa

from nn_extractor import item_type  # noqa
from nn_extractor.item_type import ItemType


class TestItemType(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_is_ndarray_type(self):
        assert item_type.is_ndarray_type(ItemType.NNTENSOR)
