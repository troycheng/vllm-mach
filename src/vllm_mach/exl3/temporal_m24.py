"""Optional SM120 Temporal K64 kernel for the two validated M24 K6 bundles."""

import importlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


def eligible(rows: int, k: int, bits: int, count: int, output_size: int) -> bool:
    return rows == 24 and k == 5120 and bits == 6 and (
        (count, output_size) in ((8, 1024), (14, 512))
    )


@lru_cache(maxsize=1)
def load_extension():
    name = "exllamav3_temporal_m24_ext"
    try:
        extension = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        logger.warning("EXL3 Temporal M24 extension is absent; using the existing BF16 path.")
        return None
    if not callable(getattr(extension, "run_grouped", None)):
        raise RuntimeError(f"{name} lacks run_grouped; rebuild it against the serving runtime.")
    return extension
