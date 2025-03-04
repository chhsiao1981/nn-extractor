# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Self, Optional, Any

from nn_extractor.ops.op_item import OpItem

from . import primitive
from .primitive import Primitive

import os
import base64

from . import cfg
from . import nnextractor_pb2

from . import item_type
from .item_type import ItemType

from .types import MetaItem, NNTensor
from .nii import NII
from . import nntensor
from . import utils

type ItemValue = NII | NNTensor | Primitive | OpItem

type RecursiveItemValue = ItemValue | list[RecursiveItemValue] | dict[str, RecursiveItemValue]


type InitItemValueType = tuple[Any, ItemType] | tuple[Any, ItemType, str] | RecursiveItemValue


@dataclass
class Item(object):
    name: str = ''

    the_type: ItemType = ItemType.UNSPECIFIED
    value: Optional[ItemValue | list[Self] | tuple[Self] | dict[str, Self]] = None

    other_type: Optional[str] = None

    data_id: Optional[str] = None

    def __init__(
        self: Self,
        name: str,
        value: InitItemValueType | list[Self] | tuple[Self] | dict[str, Self],

        the_type: Optional[ItemType] = None,

        other_type: Optional[str] = None,

        data_id: Optional[str] = None,
        item_idx: Optional[int] = None,
    ):
        if data_id is None and item_idx is not None:
            data_id = f'{item_idx}_{name}'

        if the_type is None:
            value, the_type, other_type = Item._parse_type_value(value)

        if the_type == ItemType.LIST:
            value = [Item._single_item(each, name, data_id, idx) for idx, each in enumerate(value)]
        elif the_type == ItemType.MAP:
            value = {k: Item._single_item(each, name, data_id, k) for k, each in value.items()}
        elif the_type == ItemType.OP_ITEM:
            value.set_data_id(data_id)

        self.name = name
        self.the_type = the_type
        self.value = value
        self.other_type = other_type
        self.data_id = data_id

    @classmethod
    def _parse_type_value(
        cls: Self,
        value: InitItemValueType | list[Self] | tuple[Self] | dict[str, Self],
    ) -> tuple[Any, ItemType, Optional[str]]:
        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) == 2 and isinstance(value[1], ItemType):
                return value[0], value[1], None
            elif len(value) == 3 and value[1] == ItemType.OTHER:
                return value[0], value[1], value[2]
            else:
                return value, ItemType.LIST, None
        elif isinstance(value, dict):
            return value, ItemType.MAP, None
        elif isinstance(value, NII):
            return value, ItemType.NII, None
        elif nntensor.isinstance_nntensor(value):
            return value, ItemType.NNTENSOR, None
        elif isinstance(value, OpItem):
            return value, ItemType.OP_ITEM, None
        else:
            return value, ItemType.RAW, None

    @classmethod
    def _single_item(
        cls: Self,
        value: Any,
        name: str,
        data_id: Optional[str],
        the_idx: int | str,
    ) -> Self:
        if isinstance(value, Item):
            return value

        child_name = f'{name}_{the_idx}'
        child_data_id = None if data_id is None else f'{data_id}/{the_idx}'

        return Item(value=value, name=child_name, data_id=child_data_id)

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, Item):
            return False

        if self.name != other.name:
            return False

        if self.the_type != other.the_type:
            return False

        if self.the_type == ItemType.OTHER and self.other_type != other.other_type:
            return False

        if self.the_type == ItemType.NII:
            if self.value != other.value:
                return False
        elif item_type.is_ndarray_type(self.the_type):
            if not nntensor.eq_nntensor(self.value, other.value):
                return False
        elif self.the_type == ItemType.OP_ITEM:
            if self.value != other.value:
                return False
        elif self.the_type == ItemType.LIST:
            if self.value != other.value:
                return False
        elif self.the_type == ItemType.MAP:
            if self.value != other.value:
                return False
        else:
            if not primitive.eq_primitive(self.value, other.value):
                return False

        return True

    def serialize_pb(self: Self) -> nnextractor_pb2.Item:
        if self.the_type == ItemType.NII:
            serialized_nii = self.value.serialize_pb()
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                nii=serialized_nii,
            )
        elif item_type.is_ndarray_type(self.the_type):
            serialized_tensor = nntensor.serialize_nntensor_pb(self.value)
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                tensor=serialized_tensor,
            )
        elif self.the_type == ItemType.OP_ITEM:
            serialized_op_item = self.value.serialize_pb()
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                op_item=serialized_op_item,
            )
        elif self.the_type != ItemType.OTHER:
            serialized_primitive = primitive.serialize_primitive_pb(self.value)
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                primitive=serialized_primitive,
            )
        else:
            serialized_primitive = primitive.serialize_primitive_pb(self.value)
            return nnextractor_pb2.Item(
                name=self.name,
                the_type=int(self.the_type),
                other_type=self.other_type,
                primitive=serialized_primitive,
            )

    @classmethod
    def deserialize_pb(cls: Self, item_pb: nnextractor_pb2.Item) -> Self:
        name = item_pb.name
        the_type = ItemType(item_pb.the_type)
        if the_type == ItemType.NII:
            value_nii = NII.deserialize_pb(item_pb.nii)
            return Item(
                name=name,
                the_type=the_type,
                value=value_nii,
            )
        elif item_type.is_ndarray_type(the_type):
            value_ndarray = nntensor.deserialize_nntensor_pb(item_pb.tensor)
            return Item(
                name=name,
                the_type=the_type,
                value=value_ndarray,
            )
        elif the_type == ItemType.OP_ITEM:
            value_op_item = OpItem.deserialize_pb(item_pb.op_item)
            return Item(
                name=name,
                the_type=the_type,
                value=value_op_item,
            )
        else:
            value_primitive = primitive.deserialize_primitive_pb(item_pb.primitive)
            if the_type != ItemType.OTHER:
                return Item(
                    name=name,
                    the_type=the_type,
                    value=value_primitive,
                )
            else:
                return Item(
                    name=name,
                    the_type=the_type,
                    other_type=item_pb.other_type,
                    value=value_primitive,
                )

    def meta(self: Self, is_recursive=False) -> MetaItem:
        the_type = self.other_type if self.the_type == ItemType.OTHER else str(self.the_type)
        meta_value: MetaItem = None
        data_id: Optional[str] = None
        if self.the_type == ItemType.LIST:
            meta_value = [each.meta(is_recursive=True) for each in self.value]
            if is_recursive:
                return meta_value
        elif self.the_type == ItemType.MAP:
            meta_value = {k: each.meta(is_recursive=True) for k, each in self.value.items()}
            if is_recursive:
                return meta_value
        elif self.the_type == ItemType.NII:
            value_nii: NII = self.value
            meta_value = value_nii.meta()
            data_id = self.data_id
        elif item_type.is_ndarray_type(self.the_type):
            meta_value = nntensor.meta_nntensor(self.value)
            data_id = self.data_id
        elif self.the_type == ItemType.OP_ITEM:
            meta_value = self.value.meta()
            if is_recursive:
                return meta_value
        else:
            # we can directly return meta_value because:
            # 1. now items is using item_dict to specify the data.
            # 2. primitive is restricted to single number or str, not with list or dict.
            meta_value = primitive.meta_primitive(self.value)
            if is_recursive:
                return meta_value
        return MetaItem(
            name=self.name,
            the_type=the_type,
            value=meta_value,
            data_id=data_id,
        )

    def save_to_file(self: Self, seq_dir: str):
        if self.the_type == ItemType.LIST:
            [each.save_to_file(seq_dir) for each in self.value]
            return
        elif self.the_type == ItemType.MAP:
            [each.save_to_file(seq_dir) for each in self.value.values()]
            return
        elif self.the_type == ItemType.OP_ITEM:
            self.value.save_to_file(seq_dir)
            return

        self.save_to_file_pb(seq_dir)

    def save_to_file_pb(self: Self, seq_dir: str):
        if item_type.is_skip_save_file_type(self.the_type):
            return

        serialized = self.serialize_pb()

        filename = f"{self.data_id}.pb"

        cfg.logger.info(f'to save_to_file_pb: seq_dir: {seq_dir} data_id: {self.data_id} the_type: {self.the_type}')  # noqa
        out_filename = os.sep.join([cfg.config['output_dir'], seq_dir, filename])
        if os.path.exists(out_filename):
            cfg.logger.warning(f'save_to_file_pb: (item) file already exists: seq_dir: {seq_dir} data_id: {self.data_id} out_filename: {out_filename}')  # noqa

        utils.ensure_dir(out_filename)
        with open(out_filename, 'wb') as f:
            serialized_bytes = serialized.SerializeToString()
            serialized_str = base64.b64encode(serialized_bytes)
            f.write(serialized_str)
