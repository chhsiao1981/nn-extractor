# -*- coding: utf-8 -*-


from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Self

from nn_extractor import nnextractor_pb2, primitive
from nn_extractor.types import Primitive


class RecursivePrimitiveType(Enum):
    UNSPECIFIED = nnextractor_pb2.RecursivePrimitiveType.RP_UNSPECIFIED

    PRIMITIVE = nnextractor_pb2.RecursivePrimitiveType.RP_PRIMITIVE

    PRIMITIVES = nnextractor_pb2.RecursivePrimitiveType.RP_PRIMITIVES

    PRIMITIVE_MAP = nnextractor_pb2.RecursivePrimitiveType.RP_PRIMITIVE_MAP

    SLICE = nnextractor_pb2.RecursivePrimitiveType.RP_SLICE

    def __int__(self: Self):
        return self.value

    def __str__(self: Self):
        return self.name.lower()


type RecursivePrimitiveValue = Optional[Primitive] | Optional[list[Self]] | Optional[dict[str, Self]]  # noqa


@dataclass
class RecursivePrimitive(object):
    the_type: RecursivePrimitiveType
    value: RecursivePrimitiveValue = None

    def __init__(
        self, value: Any,
        the_type: RecursivePrimitiveType = RecursivePrimitiveType.UNSPECIFIED,
    ):
        if the_type != RecursivePrimitiveType.UNSPECIFIED:
            self.value = value
            self.the_type = the_type
        elif isinstance(value, list) or isinstance(value, tuple):
            self.value = [RecursivePrimitive(each) for each in value]
            self.the_type = RecursivePrimitiveType.PRIMITIVES
        elif isinstance(value, slice):
            self.value = [value.start, value.stop, value.step]
            self.the_type = RecursivePrimitiveType.SLICE
        elif isinstance(value, dict):
            self.value = {k: RecursivePrimitive(each) for k, each in value.items()}
            self.the_type = RecursivePrimitiveType.PRIMITIVE_MAP
        elif isinstance(value, RecursivePrimitive):
            self.the_type = value.the_type
            self.value = value.value
        else:
            self.value = value
            self.the_type = RecursivePrimitiveType.PRIMITIVE

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, RecursivePrimitive):
            return False
        if self.the_type != other.the_type:
            return False

        if self.the_type == RecursivePrimitiveType.PRIMITIVE:
            return self.value == other.value
        elif self.the_type == RecursivePrimitiveType.PRIMITIVES:
            if len(self.value) != len(other.value):
                return False

            for idx, each in enumerate(self.value):
                if each != other[idx]:
                    return False
        elif self.the_type == RecursivePrimitiveType.PRIMITIVE_MAP:
            if len(self.value) != len(other.value):
                return False
            for k, each in self.value.items():
                if each != other[k]:
                    return False
        elif self.the_type == RecursivePrimitiveType.SLICE:
            if len(self.value) != len(other.value):
                return False

            for idx, each in enumerate(self.value):
                if each != other[idx]:
                    return False

        return True

    def serialize_pb(self: Self):
        if self.the_type == RecursivePrimitiveType.PRIMITIVE:
            serialized_primitive = primitive.serialize_primitive_pb(self.value)
            return nnextractor_pb2.RecursivePrimitive(
                the_type=int(self.the_type),
                primitive=serialized_primitive,
            )
        elif self.the_type == RecursivePrimitiveType.PRIMITIVES:
            serialized_primitive_map = [each.serialize_pb() for each in self.value]
            return nnextractor_pb2.RecursivePrimitive(
                the_type=int(self.the_type),
                primitives=serialized_primitive_map,
            )
        elif self.the_type == RecursivePrimitiveType.PRIMITIVE_MAP:
            serialized_primitive_map = {k: each.serialize_pb() for k, each in self.value.items()}
            return nnextractor_pb2.RecursivePrimitive(
                the_type=int(self.the_type),
                primitive_map=serialized_primitive_map,
            )
        elif self.the_type == RecursivePrimitiveType.SLICE:
            serialized_value = [primitive.serialize_primitive_pb(each) for each in self.value]
            return nnextractor_pb2.RecursivePrimitive(
                the_type=int(self.the_type),
                slice=serialized_value,
            )
        else:
            raise Exception(f'invalid recursive primtive type: the_type: {self.the_type}')

    @classmethod
    def deserialize_pb(
        cls: Self,
        recursive_primitive_pb: nnextractor_pb2.RecursivePrimitive,
    ) -> Self:
        if recursive_primitive_pb.the_type == nnextractor_pb2.RecursivePrimitiveType.RP_PRIMITIVE:
            return RecursivePrimitive(
                the_type=RecursivePrimitiveType.PRIMITIVE,
                value=recursive_primitive_pb.primitive,
            )
        elif recursive_primitive_pb.the_type == nnextractor_pb2.RecursivePrimitiveType.RP_PRIMITIVES:  # noqa
            value = [
                RecursivePrimitive.deserialize_pb(each)
                for each in recursive_primitive_pb.primitives
            ]
            return RecursivePrimitive(the_type=RecursivePrimitiveType.PRIMITIVES, value=value)
        elif recursive_primitive_pb.the_type == nnextractor_pb2.RecursivePrimitiveType.RP_PRIMITIVE_MAP:  # noqa
            value = {
                k: RecursivePrimitive.deserialize_pb(each)
                for k, each in recursive_primitive_pb.primitive_map.items()
            }
            return RecursivePrimitive(the_type=RecursivePrimitiveType.PRIMITIVE_MAP, value=value)
        elif recursive_primitive_pb.the_type == nnextractor_pb2.RecursivePrimitiveType.RP_SLICE:
            deserialized_slice = [
                primitive.deserialize_primitive_pb(each)
                for each in recursive_primitive_pb.slice
            ]
            value = slice(
                start=deserialized_slice[0],
                stop=deserialized_slice[1],
                step=deserialized_slice[2],
            )
            return RecursivePrimitive(the_type=RecursivePrimitiveType.SLICE, value=value)
        else:
            raise Exception(f'RecursivePrimitive.deserialize_pb: invalid type: {recursive_primitive_pb.the_type}')  # noqa

    def meta(self: Self):
        if self.the_type == RecursivePrimitiveType.PRIMITIVES:
            return [each.meta() for each in self.value]
        elif self.the_type == RecursivePrimitiveType.PRIMITIVE_MAP:
            return {k: each.meta() for k, each in self.value.items()}
        elif self.the_type == RecursivePrimitiveType.SLICE:
            return [primitive.meta_primitive(each) for each in self.value]
        else:
            return primitive.meta_primitive(self.value)
