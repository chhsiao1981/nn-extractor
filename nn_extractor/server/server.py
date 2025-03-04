# -*- coding: utf-8 -*-

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from nn_extractor import cfg

from . import init

from .handlers.get_model_list_handler import get_model_list_handler
from .handlers.get_model_handler import get_model_handler
from .handlers.get_meta_list_handler import get_meta_list_handler
from .handlers.get_meta_handler import get_meta_handler
from .handlers.get_protobuf_handler import get_protobuf_handler, GetProtobufBody

app = FastAPI(
    title="nn-extractor",
    root_path="/api/v1",
    on_startup=[init.init],
    debug=True,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.config['server']['origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def get_root():
    return {'version': cfg.config['version']}


@app.get('/model/list')
def get_model_list(
    start_idx: int = Query(0, description='starting index'),
    n: int = Query(100, description='number to list'),
    asc: bool = Query(True, description='is ascending'),
):
    return get_model_list_handler(start_idx, n, asc)


@app.get('/model/{model_id}')
def get_model(model_id):
    '''
    get-model
    '''
    return get_model_handler(model_id)


@app.get('/meta/list')
def get_meta_list(
    start_idx: int = Query(0, title='starting index', description='starting index'),
    n: int = Query(100, description='number to list'),
    asc: bool = Query(True, description='is ascending'),
):
    return get_meta_list_handler(start_idx, n, asc)


@app.get('/meta/{meta_id}')
def get_meta(meta_id: str):
    '''
    get-meta
    '''
    return get_meta_handler(meta_id)


@app.post('/protobuf')
def get_protobuf(body: GetProtobufBody):
    return get_protobuf_handler(body)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
