# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, TypedDict, Optional, Any, NamedTuple
from . import primitive
from .primitive import Primitive, MetaPrimitive

import os
import pickle

from . import nnextractor_pb2

from .item_type import ItemType


class MetaItem(TypedDict):
    name: str
    the_type: str
    item: MetaPrimitive


@dataclass
class ItemValue(object):
    the_type: str
    item: Primitive
    other_type: Optional[str] = None


class PickleItem(NamedTuple):
    name: str
    the_type: ItemType
    item: Primitive

    other_type: Optional[str]


@dataclass
class Item(object):
    name: str
    the_type: ItemType
    item: Primitive

    other_type: Optional[str] = None

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, Item):
            return False

        if self.name != other.name:
            return False

        if self.the_type != other.the_type:
            return False

        if self.the_type == ItemType.OTHER and self.other_type != other.other_type:
            return False

        if not primitive.eq_primitive(self.item, other.item):
            return False

        return True

    def serialize_pk(self: Self) -> PickleItem:
        return PickleItem(
            name=self.name,
            the_type=self.the_type,
            item=self.item,
            other_type=self.other_type,
        )

    @classmethod
    def deserialize_pk(cls: Self, item_pk: PickleItem) -> Self:
        return Item(
            name=item_pk.name,
            the_type=item_pk.the_type,
            item=item_pk.item,
            other_type=item_pk.other_type,
        )

    def serialize_pb(self: Self) -> nnextractor_pb2.Item:
        serialized_item = primitive.serialize_primitive_pb(self.item)
        if self.the_type != ItemType.OTHER:
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                item=serialized_item,
            )
        else:
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                other_type=self.other_type,
                item=serialized_item,
            )

    @classmethod
    def deserialize_pb(cls: Self, item_pb: nnextractor_pb2.Item) -> Self:
        item = primitive.deserialize_primitive_pb(item_pb.item)
        if item_pb.the_type != nnextractor_pb2.ItemType.I_OTHER:
            return Item(
                name=item_pb.name,
                the_type=ItemType(item_pb.the_type),
                item=item,
            )
        else:
            return Item(
                name=item_pb.name,
                the_type=ItemType(item_pb.the_type),
                other_type=item_pb.other_type,
                item=item,
            )

    def meta(self: Self) -> MetaItem:
        the_type = self.other_type if self.the_type == ItemType.OTHER else str(self.the_type)
        return MetaItem(name=self.name, the_type=the_type, item=primitive.meta_primitive(self.item))

    def save_to_file(self: Self, output_dir: str, prompt: str = ''):
        self.save_to_file_pb(output_dir, prompt)

    def save_to_file_pk(self: Self, output_dir: str, prompt: str = ''):
        serialized = self.serialize_pk()

        full_name = f"{'_'.join(list(filter(None, [prompt, self.name])))}.pk"

        out_filename = os.sep.join([output_dir, full_name])
        with open(out_filename, 'wb') as f:
            pickle.dump(serialized, f)

    def save_to_file_pb(self: Self, output_dir: str, prompt: str = ''):
        serialized = self.serialize_pb()

        full_name = f"{'_'.join(list(filter(None, [prompt, self.name])))}.pb"

        out_filename = os.sep.join([output_dir, full_name])
        with open(out_filename, 'wb') as f:
            f.write(serialized.SerializeToString())
