"""Global seeding for reproducible runs.

Every entry point calls :func:`seed_everything` exactly once, right after the
config is resolved, and records the seed in the run manifest. Torch is seeded
only if it is installed, so the reproducibility core stays torch-free.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and (if present) PyTorch.

    Parameters
    ----------
    seed:
        The seed to apply everywhere.
    deterministic:
        If True, force deterministic cuDNN/algorithm selection where the
        framework supports it. Slightly slower, but required so test-set
        numbers are bit-reproducible across machines.

    Returns
    -------
    int
        The seed that was applied (echoed for manifest logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional at E0
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt-in deterministic kernels; guarded because some ops lack them.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            except Exception:
                pass
    except ModuleNotFoundError:
        pass

    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding — derive each worker seed from the base seed."""
    base = np.random.get_state()[1][0]
    seed = int(base) + worker_id
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
