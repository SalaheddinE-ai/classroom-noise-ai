"""Test fixtures: isolated temp database and data dir (set before importing the app)."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp(prefix="cnai_test_")
os.environ["CNAI_DATA_DIR"] = _tmp
os.environ["CNAI_DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"

# --- minimal torch stand-in, used ONLY when real torch is not installed -------------------
# It lets the policy-engine / API tests run without the 2+ GB torch wheel. Tests that truly
# need torch (real VAD, audio decoding) check REAL_TORCH and skip otherwise.
import types

try:
    import torch  # noqa: F401
    REAL_TORCH = True
except Exception:
    REAL_TORCH = False
    import numpy as _np

    fake = types.ModuleType("torch")
    fake.Tensor = _np.ndarray
    fake.from_numpy = lambda a: a
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake.no_grad = fake.inference_mode = lambda: _Ctx()
    sys.modules["torch"] = fake

os.environ["CNAI_TEST_REAL_TORCH"] = "1" if REAL_TORCH else "0"
