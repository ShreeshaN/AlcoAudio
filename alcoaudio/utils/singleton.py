"""
Created on 3rd Jan 2017
@author: Shreesha N

Description:
    Singleton class
    Used to create singleton objects
"""

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]