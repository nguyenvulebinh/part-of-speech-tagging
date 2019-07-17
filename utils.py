import os
import sys
import importlib


def import_user_module(module_path):
    if module_path is not None:
        module_path = os.path.abspath(module_path)
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)
