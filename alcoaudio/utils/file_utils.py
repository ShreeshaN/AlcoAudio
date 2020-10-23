# -*- coding: utf-8 -*-
"""
@created on: 02/16/20,
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


def delete_file(path):
    if os.path.exists(path):
        os.remove(path)
