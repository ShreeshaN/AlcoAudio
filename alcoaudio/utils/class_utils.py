# -*- coding: utf-8 -*-
"""
@created on: 11/29/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
