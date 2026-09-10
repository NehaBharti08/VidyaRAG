"""Standard-library shims for the oldest Python this package must run on.

The floor is not a preference. The demo is hosted on a Hugging Face ZeroGPU
Space, and ZeroGPU images are pinned to Python 3.10 by the platform: a
`python_version` in the Space README is accepted into the repo and then ignored
by the builder, which still pulls `python:3.10.13`. Code that ships to the Space
therefore has to import cleanly on 3.10 no matter what the development
environment runs.

Two 3.11 additions were being used, both with exact older equivalents:

* `enum.StrEnum` -- `str` mixed into `enum.Enum` gives the same behaviour, with
  `__str__` restored to `str.__str__` so members format as their value rather
  than as `Class.MEMBER`. That formatting difference is the one that silently
  corrupts output rather than raising, which is why it is set explicitly.
* `datetime.UTC` -- a plain alias for `datetime.timezone.utc` upstream.

Import both from here rather than from the standard library directly. There is
a CI job on 3.10 that fails if a bare 3.11 spelling comes back.
"""

from __future__ import annotations

import enum
import sys
from datetime import timezone

__all__ = ["UTC", "StrEnum"]

UTC = timezone.utc

if sys.version_info >= (3, 11):
    StrEnum = enum.StrEnum
else:

    class StrEnum(str, enum.Enum):
        """Backport of `enum.StrEnum` for Python 3.10.

        `enum.auto()` is deliberately not supported. Upstream `StrEnum` lowers
        the member name to produce an auto value, and reimplementing that here
        would create a rule that holds on one Python and not the other. Every
        member in this package is given an explicit value, so the two spellings
        cannot drift; a future `auto()` will raise on 3.10 rather than quietly
        yield a different value than it does on 3.11.
        """

        __str__ = str.__str__
