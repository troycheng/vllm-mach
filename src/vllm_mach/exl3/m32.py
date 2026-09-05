"""Optional exact-M32 grouped EXL3 extension."""

import importlib
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_extension():
    try:
        extension = importlib.import_module("exllamav3_m32_ext")
    except ModuleNotFoundError as exc:
        if exc.name != "exllamav3_m32_ext":
            raise
        logger.warning("EXL3 M32 extension is absent; using M16+M16.")
        return None
    if not callable(getattr(extension, "grouped_had_m32", None)):
        raise RuntimeError("exllamav3_m32_ext lacks grouped_had_m32; rebuild it.")
    logger.info("EXL3 BF16-I/O true TILE_M32 extension loaded.")
    return extension
