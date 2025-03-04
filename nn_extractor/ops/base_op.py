# -*- coding: utf-8 -*-


from dataclasses import dataclass
from typing import Self


@dataclass
class BaseOp(object):
    def integrate(self: Self, name: str):
        raise Exception('not implemented')
