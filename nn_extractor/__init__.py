# -*- coding: utf-8 -*-

from .hook import register_hook
from .hook import register_forward_hook
from .hook import register_backward_hook

from .save_file import save_file

from .activations import serialize_activations
from .activations import deserialize_activations

from .activation import serialize_activation
from .activation import deserialize_activation

from .parameter import serialize_parameter
from .parameter import deserialize_parameter

from .record import serialize_record
from .record import deserialize_record

from .ndarray import serialize_ndarray
from .ndarray import deserialize_ndarray
