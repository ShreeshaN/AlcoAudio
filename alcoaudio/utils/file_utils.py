# -*- coding: utf-8 -*-
"""
@created on: 11/29/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import os
import shutil


def create_dirs(paths):
    for path in paths:
        if not os.path.exists(path):
            print('Creating folder - ', path)
            os.makedirs(path)


def cp_file(src, dst):
    print('Copying data from', src, 'to', dst)
    shutil.copyfile(src, dst)
