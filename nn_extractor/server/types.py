# -*- coding: utf-8 -*-


from typing import Any, TypedDict, Optional, Self

from nn_extractor.types import MetaNNExtractor, MetaNNRecord, MetaNNParameter, MetaItems

from pydantic import BaseModel, Field


class ModelSummary(TypedDict):
    name: str
    classname: str
    default_params: dict[str, Any]


class Model(ModelSummary):
    the_class: type


class Meta(TypedDict):
    name: str
    meta: MetaNNExtractor


class MetaNNNodeSummary(TypedDict):
    name: str

    inputs: list[MetaNNRecord]
    params: list[MetaNNParameter]
    activation: Optional[MetaNNRecord]

    gradient_inputs: list[MetaNNRecord]
    gradient_params: list[MetaNNParameter]
    gradients: list[MetaNNRecord]

    children: Optional[list[Self] | int]


class MetaTaskflowSummary(TypedDict):
    name: str
    flow_type: str
    flow_id: int


class MetaNNExtractorSummary(TypedDict):
    name: str

    taskflow: int
    inputs: list[MetaItems]
    preprocesses: list[MetaItems]
    nodes: list[MetaNNNodeSummary]
    postprocesses: list[MetaItems]
    outputs: list[MetaItems]

    extractors: list[Self] | int


class MetaSummary(TypedDict):
    name: str
    meta: MetaNNExtractorSummary
