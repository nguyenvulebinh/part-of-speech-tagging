import os
import sys
import importlib
import re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line_word(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_line_char(line):
    line = SPACE_NORMALIZER.sub("", line)
    line = line.strip()
    return list(line)


def import_user_module(module_path):
    if module_path is not None:
        module_path = os.path.abspath(module_path)
        module_parent, module_name = os.path.split(module_path)

        if module_name not in sys.modules:
            sys.path.insert(0, module_parent)
            importlib.import_module(module_name)
            sys.path.pop(0)
