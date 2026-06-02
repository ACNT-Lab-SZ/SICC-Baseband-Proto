#!/usr/bin/env python3
from __future__ import annotations

import inspect

import teraboxdl


print(teraboxdl.__file__)
print(dir(teraboxdl))
try:
    print(inspect.getsource(teraboxdl)[:3000])
except Exception as exc:
    print(repr(exc))
