# -*- coding: utf-8 -*-

from nn_extractor import argparse
from pydantic import BaseModel, Field
import json

from .server import app


class Args(BaseModel):
    filename: str = Field(
        alias='f',
        default='openapi.json',
        description='output filename')

    is_print: bool = Field(
        action='store_true',
        default=False,
        description='print on screen')


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    argparse.add_args(parser, Args())

    return parser.parse_args()


def main():
    args = parse_args()
    openapi_dict = app.openapi()
    openapi_json = json.dumps(openapi_dict, indent=2)

    with open(args.filename, 'w') as f:
        f.write(openapi_json)

    if args.is_print:
        print(openapi_json)


if __name__ == '__main__':
    main()
