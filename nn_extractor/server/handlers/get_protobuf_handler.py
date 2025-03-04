# -*- coding: utf-8 -*-

from fastapi import HTTPException
from pydantic import BaseModel, Field

import os

from nn_extractor import cfg
from nn_extractor.taskflow import TaskflowType


class GetProtobufBody(BaseModel):
    extractor_id: str = Field(description='extractor-id (name) to get the content, if it is from a sub-extractor, use the id of the sub-extractor')  # noqa
    flow_type: int = Field(description='sequence type. refer to protobuf definition about the SequenceType')  # noqa
    flow_id: str = Field(description='seq-id in the sequence, not the index in the sequence-list, but the specific seq_id within the Sequence class')  # noqa
    data_id: str = Field(description='data_id to get the content')


class ProtobufResponse(BaseModel):
    bytes: str = Field(default='', description='base64 encoded protobuf')
    errmsg: str = Field(default='', description='errmsg')


def get_protobuf_handler(body: GetProtobufBody):

    extractor_id = body.extractor_id
    flow_type = TaskflowType(body.flow_type)
    flow_id = body.flow_id
    data_id = body.data_id

    root_dir = cfg.config.get('server', {}).get('root_dir', '')
    if not root_dir:
        raise HTTPException(status_code=500, detail='invalid server config')

    filename = os.sep.join(
        [root_dir, f'{extractor_id}', f'{flow_type}', f'{flow_id}', f'{data_id}.pb']
    )

    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail='file not found')

    with open(filename, 'r') as f:
        the_bytes = f.read()

    return ProtobufResponse(bytes=the_bytes)
