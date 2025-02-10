# -*- coding: utf-8 -*-

from typing import Optional
from argparse import ArgumentParser

import re
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined


def add_args(parser: ArgumentParser, args: BaseModel):
    '''
    Add args (as a Pydantic model) to an ArgumentParser

    The followings are the differences compared to the original ArgumentParser:

    * instead of setting the dest name to the full-name, the member variable name
      is always the dest name (can be alias or full-name).
    * can be dashed argument or underscored argument if specified any.
    * by default, prefixed with '-' if the field name is single letter (`alias`),
      and prefixed with '--' if the field name is >= 2 letters (`full_name`).
    * `alias` and `full_name` can also be specified as part of the `Field` parmeters.

    https://stackoverflow.com/questions/72741663/argument-parser-from-a-pydantic-model

    now all the field info belongs to either basic attributes (ex: description)
    or json_schema_extra (ex: nargs) or metadata (ex: gt)
    https://github.com/pydantic/pydantic/blob/main/pydantic/fields.py#L186

    now type_ becomes field.annotation
    https://github.com/pydantic/pydantic/blob/main/pydantic/fields.py#L97
    '''

    fields = args.model_fields
    for name, field in fields.items():
        names, dest_name = _parse_names(name, field)

        action = _parse_action(field)
        the_type = _parse_type(field, action)
        nargs = _parse_nargs(field)
        required = _parse_required(field, action, nargs)

        kwargs = {
            'default': field.default,
            'help': field.description,
            'required': required,
            'action': action,
        }
        if the_type is not None:
            kwargs['type'] = the_type
        if nargs is not None:
            kwargs['nargs'] = nargs

        parser.add_argument(
            *names,
            dest=dest_name,
            **kwargs,
        )


def _parse_names(name: str, field: FieldInfo) -> tuple[list[str], str]:
    alias = ''
    full_name = ''
    full_dashed_name = ''
    full_underscore_name = ''

    if len(name) == 1:
        alias = name
    else:
        full_name = name

    field_short_name = getattr(field, 'alias', '')
    if field_short_name:
        alias = field_short_name

    field_full_name = getattr(field, 'full_name', '')
    if field_full_name:
        full_name = field_full_name

    full_dashed_name = re.sub(r'_', '-', full_name)
    full_underscore_name = re.sub(r'-', '_', full_dashed_name)

    dest_name = name

    names = []
    if alias:
        names.append(f'-{alias}')
    if full_dashed_name:
        names.append(f'--{full_dashed_name}')
    if full_underscore_name:
        names.append(f'--{full_underscore_name}')

    return names, dest_name


def _parse_action(field: FieldInfo) -> Optional[str]:
    if field.json_schema_extra is None:
        return None

    return field.json_schema_extra.get('action', None)


def _parse_required(field: FieldInfo, action: str, nargs: str) -> Optional[bool]:
    ''''
    determine required based on field values.

    We check:
    1. if required is specified, then return required.
    2. otherwise, required if field.default == PydanticUndefined.
    '''
    if action in ['store_true', 'store_false']:
        return None
    if nargs:
        return None

    if field.json_schema_extra is None:
        return None

    if 'required' in field.json_schema_extra:
        return field.json_schema_extra['required']

    return field.default == PydanticUndefined


def _parse_type(field: FieldInfo, action: str) -> Optional[str]:
    if action in ['store_true', 'store_false']:
        return None

    return field.annotation


def _parse_nargs(field: FieldInfo) -> Optional[str]:
    if field.json_schema_extra is None:
        return None

    return field.json_schema_extra.get('nargs', None)
