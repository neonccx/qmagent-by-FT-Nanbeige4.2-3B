"""Scoped NumPy allowlist for locally generated Trainer RNG checkpoints."""
from contextlib import contextmanager


@contextmanager
def numpy_rng_loading():
    import numpy as np
    import torch
    # No weights_only=False and no process-wide persistent allowlist.
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32")),
    ]):
        yield
