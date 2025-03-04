# -*- coding: utf-8 -*-

import os


def ensure_dir(filename: str):
    the_dirname = os.path.dirname(filename)
    if os.path.exists(the_dirname):
        return
    os.makedirs(the_dirname, exist_ok=True)
