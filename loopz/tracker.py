"""
tracker.py — Core persistence engine for loopz.

Handles:
  - Loop progress (JSON)
  - ML state: PyTorch model/optimizer/scheduler/scaler,
              DataParallel, DistributedDataParallel,
              Numpy arrays, Sklearn models, plain Python objects
  - Loop variables (FIX 2)
  - Full random seed state: Python / Numpy / PyTorch / CUDA (FIX 1)
  - Corrupted file recovery
  - Atomic writes (no corrupt-on-crash)
"""

import os
import json
import time
import random
import hashlib
import pickle
import tempfile
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

CACHE_DIR = Path.home() / ".loopz"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_name(job_name: str) -> str:
    """Unique 12-char hex derived from the full job name.
    Used as the ENTIRE filename stem so special chars (/ \\ : * ? < > |)
    in job_name never corrupt the file path."""
    return hashlib.md5(job_name.encode()).hexdigest()[:12]


def _get_path(job_name: str, ext: str, base_dir: Optional[Path] = None) -> Path:
    """Return a safe flat path inside base_dir (or CACHE_DIR if not given).
    base_dir is resolved ONCE by the caller (decorator or public API) —
    internal tracker functions never need to know about checkpoint_dir.
    """
    base = base_dir if base_dir is not None else CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"loopz_{_safe_name(job_name)}{ext}"


def _atomic_pickle(path: Path, obj: Any):
    """Write to a temp file then rename — prevents partial writes on crash."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _atomic_json(path: Path, obj: dict):
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Progress — JSON (human-readable, easy to inspect)
# ---------------------------------------------------------------------------

# ✅ CHANGED: added base_dir parameter, passed to _get_path
def save_progress(
    job_name: str,
    index: int,
    total: int,
    meta: Optional[Dict] = None,
    base_dir: Optional[Path] = None,
):
    data = {
        "job_name": job_name,
        "index":    index,
        "total":    total,
        "percent":  round((index / total) * 100, 2) if total else 0.0,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta":     meta or {},
    }
    _atomic_json(_get_path(job_name, ".json", base_dir), data)


# ✅ CHANGED: added base_dir parameter, passed to _get_path
def load_progress(job_name: str, base_dir: Optional[Path] = None) -> Optional[Dict]:
    path = _get_path(job_name, ".json", base_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupted file — delete and return None (graceful recovery)
        try:
            path.unlink()
        except Exception:
            pass
        return None


# ✅ CHANGED: added base_dir parameter, passed to _get_path
def clear_progress(job_name: str, base_dir: Optional[Path] = None):
    """Remove ALL saved data for this job (progress + state + vars)."""
    for ext in [".json", ".state", ".vars", ".tmp"]:
        p = _get_path(job_name, ext, base_dir)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


# ✅ CHANGED: added base_dir parameter, uses it instead of hardcoded CACHE_DIR
def list_jobs(base_dir: Optional[Path] = None) -> List[Dict]:
    target = base_dir if base_dir is not None else CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in sorted(target.glob("loopz_*.json")):
        try:
            with open(f, encoding="utf-8") as fp:
                jobs.append(json.load(fp))
        except Exception:
            continue
    return jobs


# ---------------------------------------------------------------------------
# FIX 1 — Full Random State (Python + Numpy + PyTorch + CUDA)
# ---------------------------------------------------------------------------

def save_random_state() -> Dict:
    """Snapshot the full random state from every framework in use."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
    }
    try:
        import torch
        state["torch"] = torch.get_rng_state().clone()
        if torch.cuda.is_available():
            state["torch_cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    except (ImportError, Exception):
        pass
    return state


def restore_random_state(state: Optional[Dict]):
    """Restore full random state from a previously saved snapshot."""
    if not state:
        return
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        import torch
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (ImportError, Exception):
        pass


# ---------------------------------------------------------------------------
# FIX 2 — Loop variables (running_loss, best_acc, step counters, etc.)
# ---------------------------------------------------------------------------

# ✅ CHANGED: added base_dir parameter, passed to _get_path
def save_loop_vars(job_name: str, loop_vars: Dict, base_dir: Optional[Path] = None):
    """Persist plain Python variables that live inside the loop."""
    if not loop_vars:
        return
    try:
        _atomic_pickle(_get_path(job_name, ".vars", base_dir), loop_vars)
    except Exception as e:
        print(f"⚠️  loopz: Could not save loop_vars — {e}")


# ✅ CHANGED: added base_dir parameter, passed to _get_path
def load_loop_vars(job_name: str, base_dir: Optional[Path] = None) -> Optional[Dict]:
    path = _get_path(job_name, ".vars", base_dir)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        return None


# ✅ CHANGED: added base_dir parameter, passed to _get_path
def clear_loop_vars(job_name: str, base_dir: Optional[Path] = None):
    p = _get_path(job_name, ".vars", base_dir)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# State — ML objects (model, optimizer, scheduler, numpy, sklearn, etc.)
# ---------------------------------------------------------------------------

# ✅ CHANGED: added base_dir parameter, passed to _get_path
def save_state(job_name: str, state: Dict, base_dir: Optional[Path] = None):
    """
    Serialize and save all ML objects in `state`.
    Always snapshots random state alongside model weights.
    Uses atomic write — safe to call during training.
    """
    if not state:
        return
    serialized: Dict[str, Any] = {}
    for key, obj in state.items():
        try:
            serialized[key] = _serialize_obj(obj)
        except Exception as e:
            print(f"⚠️  loopz: Could not save state['{key}'] — {e}")
    # FIX 1 — always bundle random state with model state
    serialized["__random_state__"] = save_random_state()
    _atomic_pickle(_get_path(job_name, ".state", base_dir), serialized)


# ✅ CHANGED: added base_dir parameter, passed to _get_path
def load_state(job_name: str, state: Dict, base_dir: Optional[Path] = None) -> bool:
    """
    Restore all ML objects in `state` IN PLACE.
    Returns True if successfully loaded, False otherwise.
    """
    path = _get_path(job_name, ".state", base_dir)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            serialized = pickle.load(f)
    except Exception as e:
        print(f"⚠️  loopz: Could not read state file — {e}")
        return False

    # FIX 1 — restore random state first
    if "__random_state__" in serialized:
        restore_random_state(serialized["__random_state__"])

    all_ok = True
    for key, saved_data in serialized.items():
        if key == "__random_state__":
            continue
        if key not in state:
            continue
        try:
            _deserialize_into(state[key], saved_data)
        except Exception as e:
            print(f"⚠️  loopz: Could not restore state['{key}'] — {e}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Serialization — type-aware, handles every ML framework
# ---------------------------------------------------------------------------

def _serialize_obj(obj: Any) -> Dict:
    """
    Detect object type and serialize correctly.
    Covers: PyTorch (Module, DataParallel, DDP, Optimizer, Scheduler,
            GradScaler, Tensor), Numpy ndarray, Sklearn estimator,
            any picklable Python object.
    """
    # ---- PyTorch ----
    try:
        import torch
        import torch.nn as nn

        # FIX 3 — DataParallel: save the inner module only
        if isinstance(obj, nn.DataParallel):
            return {
                "__type__": "torch_model_dp",
                "data": {k: v.cpu().clone() for k, v in obj.module.state_dict().items()},
            }

        # FIX 3 — DistributedDataParallel
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(obj, DDP):
                return {
                    "__type__": "torch_model_ddp",
                    "data": {k: v.cpu().clone() for k, v in obj.module.state_dict().items()},
                }
        except (ImportError, Exception):
            pass

        # Regular nn.Module
        if isinstance(obj, nn.Module):
            return {
                "__type__": "torch_model",
                "data": {k: v.cpu().clone() for k, v in obj.state_dict().items()},
            }

        # Optimizer
        if isinstance(obj, torch.optim.Optimizer):
            return {"__type__": "torch_optimizer", "data": obj.state_dict()}

        # LR Scheduler — duck-typed (covers every scheduler)
        if hasattr(obj, "state_dict") and hasattr(obj, "last_epoch"):
            return {"__type__": "torch_scheduler", "data": obj.state_dict()}

        # GradScaler (mixed precision)
        # Supports both old API (torch.cuda.amp.GradScaler)
        # and new API (torch.amp.GradScaler) introduced in PyTorch 2.x
        _is_scaler = False
        try:
            from torch.cuda.amp import GradScaler as _CudaScaler
            if isinstance(obj, _CudaScaler):
                _is_scaler = True
        except (ImportError, Exception):
            pass
        if not _is_scaler:
            try:
                from torch.amp import GradScaler as _AmpScaler
                if isinstance(obj, _AmpScaler):
                    _is_scaler = True
            except (ImportError, Exception):
                pass
        if _is_scaler:
            return {"__type__": "torch_scaler", "data": obj.state_dict()}

        # Raw Tensor
        if isinstance(obj, torch.Tensor):
            return {"__type__": "torch_tensor", "data": obj.detach().cpu().clone()}

    except ImportError:
        pass

    # ---- Numpy ----
    if isinstance(obj, np.ndarray):
        return {"__type__": "numpy", "data": obj.copy()}

    # ---- Sklearn ----
    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        try:
            return {"__type__": "sklearn", "data": pickle.dumps(obj)}
        except Exception:
            pass

    # ---- Fallback: generic pickle ----
    try:
        return {"__type__": "pickle", "data": pickle.dumps(obj)}
    except Exception as e:
        raise RuntimeError(
            f"loopz: Object of type {type(obj).__name__} is not serializable. "
            f"Original error: {e}"
        )


def _deserialize_into(obj: Any, saved: Dict):
    """
    Restore saved_data back into obj IN PLACE.
    Every branch is explicit — no silent no-ops.
    """
    t = saved.get("__type__")

    # ---- PyTorch models (including DataParallel / DDP) ----
    if t in ("torch_model", "torch_model_dp", "torch_model_ddp"):
        try:
            import torch
            import torch.nn as nn
            target = obj
            # Unwrap DataParallel / DDP
            if isinstance(obj, nn.DataParallel):
                target = obj.module
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP
                if isinstance(obj, DDP):
                    target = obj.module
            except (ImportError, Exception):
                pass
            # Move weights to the device the model currently lives on
            try:
                device = next(target.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            state_dict = {k: v.to(device) for k, v in saved["data"].items()}
            target.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"loopz: torch model restore failed — {e}") from e

    # ---- Optimizer ----
    elif t == "torch_optimizer":
        try:
            obj.load_state_dict(saved["data"])
        except Exception as e:
            raise RuntimeError(f"loopz: optimizer restore failed — {e}") from e

    # ---- LR Scheduler ----
    elif t == "torch_scheduler":
        try:
            obj.load_state_dict(saved["data"])
        except Exception as e:
            raise RuntimeError(f"loopz: scheduler restore failed — {e}") from e

    # ---- GradScaler ----
    elif t == "torch_scaler":
        try:
            obj.load_state_dict(saved["data"])
        except Exception as e:
            raise RuntimeError(f"loopz: GradScaler restore failed — {e}") from e

    # ---- Torch Tensor ----
    elif t == "torch_tensor":
        try:
            import torch
            obj.data = saved["data"].to(obj.device)
        except Exception:
            obj.data = saved["data"]

    # ---- Numpy ----
    elif t == "numpy":
        try:
            obj[:] = saved["data"]
        except Exception as e:
            raise RuntimeError(f"loopz: numpy array restore failed — {e}") from e

    # ---- Sklearn ----
    elif t == "sklearn":
        try:
            restored = pickle.loads(saved["data"])
            # Copy all attributes in-place so the caller's reference updates
            obj.__dict__.update(restored.__dict__)
        except Exception as e:
            raise RuntimeError(f"loopz: sklearn model restore failed — {e}") from e

    # ---- Generic pickle ----
    elif t == "pickle":
        try:
            restored = pickle.loads(saved["data"])
            # In-place mutation strategies ordered by type:
            if isinstance(obj, list):
                # list supports slice assignment
                obj[:] = restored
            elif isinstance(obj, dict):
                # dict supports clear + update
                obj.clear()
                obj.update(restored)
            elif isinstance(obj, np.ndarray):
                # 0-d or mismatched shape numpy — fall back to slice
                obj[()] = restored
            elif hasattr(obj, "__dict__"):
                # Custom object — copy attributes in-place
                obj.__dict__.update(restored.__dict__)
            else:
                # True primitive (int, float, str, bool) — Python cannot
                # rebind the caller's variable from inside a function.
                # This is a Python language limitation, not a loopz bug.
                print(
                    f"⚠️  loopz: cannot restore primitive type "
                    f"'{type(obj).__name__}' in-place (Python limitation). "
                    f"Wrap it in a list: my_val = [0.0] and access as my_val[0]."
                )
        except Exception as e:
            raise RuntimeError(f"loopz: object restore failed — {e}") from e

    else:
        raise RuntimeError(f"loopz: unknown serialization type '{t}'")