"""Tiny shared helpers used by artifact dataclasses."""

import time
import uuid


def new_id() -> str:
    return str(uuid.uuid4())[:12]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
