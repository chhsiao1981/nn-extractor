# -*- coding: utf-8 -*-

from typing import Optional

from nn_extractor import cfg

import os
import json
from tqdm import tqdm

from nn_extractor.server.types import Meta
from nn_extractor.server.types import MetaSummary
from nn_extractor.server.types import MetaNNExtractorSummary
from nn_extractor.server.types import MetaNNNodeSummary
from nn_extractor.server.types import MetaTaskflowSummary
from nn_extractor.types import MetaNNExtractor, MetaNNNode, MetaTaskflow


def init_meta():
    root_dir = cfg.config.get('server', {}).get('root_dir', '')
    if not root_dir:
        return

    extractors = os.listdir(root_dir)

    meta_list = [_get_meta(each, root_dir) for each in tqdm(extractors)]

    meta_list: list[Meta] = list(filter(None, meta_list))
    meta_map = {each['name']: each['meta'] for each in meta_list}

    meta_summary_list = list(map(_meta_summary, meta_list))
    meta_summary_map = {each['name']: each['meta'] for each in meta_summary_list}

    cfg.logger.info(f'loaded meta: ({meta_map.keys()} / {len(meta_map)}')

    cfg.GLOBALS['META_LIST'] = meta_list
    cfg.GLOBALS['META_MAP'] = meta_map

    cfg.GLOBALS['META_SUMMARY_LIST'] = meta_summary_list
    cfg.GLOBALS['META_SUMMARY_MAP'] = meta_summary_map


def _get_meta(extractor: str, root_dir: str) -> Optional[Meta]:
    filename = os.sep.join([root_dir, extractor, f'{extractor}.meta.json'])
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        the_struct: MetaNNExtractor = json.load(f)

    return Meta(name=extractor, meta=the_struct)


def _meta_summary(meta: Meta) -> MetaSummary:
    return MetaSummary(
        name=meta['name'],
        meta=_meta_nn_extractor_summary(meta['meta']),
    )


def _meta_nn_extractor_summary(meta: MetaNNExtractor, layer=0) -> MetaNNExtractorSummary:
    taskflow = list(map(_meta_summary_taskflow, meta['taskflow']))

    nodes = list(map(_meta_summary_node, meta['nodes']))

    extractors = len(meta) if layer == 1 \
        else list(map(lambda each: _meta_nn_extractor_summary(each, layer + 1), meta['extractors']))

    return MetaNNExtractorSummary(
        name=meta['name'],
        taskflow=taskflow,
        inputs=meta['inputs'],
        preprocesses=meta['preprocesses'],
        nodes=nodes,
        postprocesses=meta['postprocesses'],
        outputs=meta['outputs'],
        extractors=extractors,
    )


def _meta_summary_taskflow(flow: MetaTaskflow) -> MetaTaskflowSummary:
    return MetaTaskflowSummary(
        name=flow['name'],
        flow_type=flow['the_type'],
        flow_id=flow['flow_id'],
    )


def _meta_summary_node(node: MetaNNNode, layer: int = 0) -> MetaNNNodeSummary:
    children = len(node['children']) if layer == 1 \
        else list(map(lambda each: _meta_summary_node(each, layer + 1), node['children']))

    return MetaNNNodeSummary(
        name=node['name'],
        inputs=node['inputs'],
        params=node['params'],
        activation=node['activation'],
        gradient_inputs=node['gradient_inputs'],
        gradient_params=node['gradient_params'],
        gradients=node['gradients'],
        children=children,
    )
