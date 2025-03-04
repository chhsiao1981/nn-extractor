# -*- coding: utf-8 -*-

from typing import Self, Optional, Any

from dataclasses import dataclass

from nn_extractor.ops.base_op import BaseOp

from .item import Item
from . import nnextractor_pb2

from .types import MetaItems


@dataclass
class Items(object):
    name: str
    items: list[Item]

    def __init__(
        self: Self,
        name: str = '',
        items: Optional[list[Item]] = None,
        item_dict: Optional[dict[str, Any]] = None,
    ):
        '''
        Items

        initialized either with items (pre-compiled items) or item_dict
        (name, value and the type is inferred from value).

        items has higher priority if accidentally presented both.
        '''
        self.name = name

        if items is not None:
            self.items = items
            return

        elif item_dict is not None:
            item_keys = list(item_dict.keys())
            self.items = [
                self._parse_item(idx, name, item_dict[name])
                for idx, name in enumerate(item_keys)
            ]
            return

    def _parse_item(self: Self, idx: int, name: str, value: Any) -> Item:
        sanitized_value = self._sanitize_item_value(name, value)

        return Item(name=name, value=sanitized_value, item_idx=idx)

    def _sanitize_item_value(self: Self, name: str, value: Any) -> Any:
        # special types
        if isinstance(value, slice):
            return [value.start, value.stop, value.step]

        # recursively sanitize special types.
        if isinstance(value, list) or isinstance(value, tuple):
            return [
                self._sanitize_item_value(f'{self.name}_{idx}', each)
                for idx, each in enumerate(value)
            ]
        elif isinstance(value, dict):
            return {
                k: self._sanitize_item_value(f'{self.name}_{k}', each)
                for k, each in value.items()
            }
        elif isinstance(value, BaseOp):
            return value.integrate(name)
        else:
            return value

    def serialize_pb(self: Self) -> nnextractor_pb2.Items:
        serialized_items = [each.serialize_pb() for each in self.items]
        return nnextractor_pb2.Items(name=self.name, items=serialized_items)

    @classmethod
    def deserialize_pb(cls: Self, items_pb: nnextractor_pb2.Items) -> Self:
        items = [Item.deserialize_pb(each) for each in items_pb.items]
        return Items(name=items_pb.name, items=items)

    def meta(self: Self) -> MetaItems:
        return MetaItems(name=self.name, items=[each.meta() for each in self.items])

    def save_to_file(self: Self, seq_dir: str):
        [each.save_to_file(seq_dir) for each in self.items]
