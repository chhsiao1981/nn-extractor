# -*- coding: utf-8 -*-

from typing import Self, Optional, Any, TypedDict, NamedTuple

from dataclasses import dataclass

import os
import numpy as np
import torch

from .item import Item, ItemValue, MetaItem
from .item_type import ItemType
from .primitive import Primitive
from . import nnextractor_pb2


class MetaItems(TypedDict):
    name: str
    items: list[MetaItem]


class PickItems(NamedTuple):
    name: str
    items: list[Item]


@dataclass
class Items(object):
    name: str
    items: list[Item]

    def __init__(
            self: Self,
            name: str = '',
            items: Optional[list[Item]] = None,
            item_dict: Optional[dict[str, Any]] = None):
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
            self.items = [self._parse_item(name, value) for name, value in item_dict.items()]
            return

    def _parse_item(self: Self, name: str, value: Any) -> Item:
        sanitized_value = self._sanitize_item_value(value)
        the_type = self._parse_item_type(sanitized_value)
        if the_type is None:
            raise Exception(f'invalid type: name: {name} value: ({value} / {type(value)}) sanitized_value: ({sanitized_value} / {type(sanitized_value)})')  # noqa

        item: Primitive = sanitized_value

        return Item(name=name, the_type=the_type, item=item)

    def _sanitize_item_value(self: Self, value: Any) -> Primitive:
        # special types
        if isinstance(value, torch.Tensor):
            return value.detach().to('cpu').numpy()
        elif isinstance(value, slice):
            return [value.start, value.stop, value.step]

        if isinstance(value, list) or isinstance(value, tuple):
            return [self._sanitize_item_value(each) for each in value]
        elif isinstance(value, dict):
            return {k: self._sanitize_item_value(each) for k, each in value.items()}
        else:
            return value

    def _parse_item_type(self: Self, value: Any) -> Optional[ItemType]:
        # already item: directly return
        if isinstance(value, Item):
            return value.the_type

        # already item-value: directly return
        if isinstance(value, ItemValue):
            return value.the_type

        # expecting Primitive
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                return ItemType.NII
            elif value.ndim == 2:
                return ItemType.IMAGE
            else:
                return ItemType.NDARRAY
        elif isinstance(value, str):
            return ItemType.TEXT
        elif isinstance(value, bool | float | int | np.bool | np.float64 | np.float32 | np.int64 | np.int32):
            return ItemType.NUMBER
        elif value is None:
            return ItemType.NULL
        elif isinstance(value, list) or isinstance(value, tuple):
            if not value:
                return ItemType.RAW
            else:
                return self._parse_item_type(value[0])
        elif isinstance(value, dict):
            if not value:
                return ItemType.RAW
            else:
                for _, v in value.items():
                    return self._parse_item_type(v)
        else:
            return None

    def serialize_pk(self: Self) -> PickItems:
        serialized_items = [each.serialize_pk() for each in self.items]
        return nnextractor_pb2.Items(name=self.name, items=serialized_items)

    @classmethod
    def deserialize_pk(cls: Self, items_pk: PickItems) -> Self:
        items = [Item.deserialize_pk(each) for each in items_pk.items]
        return Items(name=items_pk.name, items=items)

    def serialize_pb(self: Self) -> nnextractor_pb2.Items:
        serialized_items = [each.serialize_pb() for each in self.items]
        return nnextractor_pb2.Items(name=self.name, items=serialized_items)

    @classmethod
    def deserialize_pb(cls: Self, items_pb: nnextractor_pb2.Items) -> Self:
        items = [Item.deserialize_pb(each) for each in items_pb.items]
        return Items(name=items_pb.name, items=items)

    def meta(self: Self) -> MetaItems:
        return MetaItems(name=self.name, items=[each.meta() for each in self.items])

    def save_to_file(self: Self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for idx, item in enumerate(self.items):
            item.save_to_file(output_dir, str(idx))
