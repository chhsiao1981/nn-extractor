# -*- coding: utf-8 -*-

import unittest

import copy

from nn_extractor import cfg


class TestCfg(unittest.TestCase):
    orig_config: dict = {}

    def setUp(self):
        self.orig_config = copy.deepcopy(cfg.config)

    def tearDown(self):
        pass

    def test_ini(self):
        cfg.init()

    def test_ini_2(self):
        filename = 'config.test.toml'

        cfg.init(filename)

        cfg.logger.info('test-info')
        cfg.logger.warning('test-warning')
        cfg.logger.debug('test-debug')
