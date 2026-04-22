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

def _get_base_dir(checkpoint_dir: Optional[str] = None) -> Path:
    base = Path(checkpoint_dir) if checkpoint_dir else CACHE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_name(job_name: str) -> str:
    return hashlib.md5(job_name.encode()).hexdigest()[:12]


def _get_path(job_name: str, ext: str, checkpoint_dir: Optional[str] = None) -> Path:
    base = _get_base_dir(checkpoint_dir)
    return base / f"loopz_{_safe_name(job_name)}{ext}"


def _atomic_pickle(path: Path, obj: Any):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _atomic_json(path: Path, obj: dict):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Progress — JSON (human-readable, easy to inspect)
# ---------------------------------------------------------------------------

def save_progress(
    job_name: str,
    index: int,
    total: int,
    meta: Optional[Dict] = None,
    checkpoint_dir: Optional[str] = None,   # ✅ ADDED
):
    data = {
        "job_name": job_name,
        "index":    index,
        "total":    total,
        "percent":  round((index / total) * 100, 2) if total else 0.0,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta":     meta or {},
    }
    _atomic_json(_get_path(job_name, ".json", checkpoint_dir), data)


def load_progress(job_name: str, checkpoint_dir: Optional[str] = None) -> Optional[Dict]:   # ✅ ADDED
    path = _get_path(job_name, ".json", checkpoint_dir)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        return None


def clear_progress(job_name: str, checkpoint_dir: Optional[str] = None):   # ✅ ADDED
    """Remove ALL saved data for this job (progress + state + vars)."""
    for ext in [".json", ".state", ".vars", ".tmp"]:
        p = _get_path(job_name, ext, checkpoint_dir)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def list_jobs(checkpoint_dir: Optional[str] = None) -> List[Dict]:   # ✅ ADDED (bonus)
    base = _get_base_dir(checkpoint_dir)
    jobs = []
    for f in sorted(base.glob("*.json")):
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
# FIX 2 — Loop variables
# ---------------------------------------------------------------------------

def save_loop_vars(job_name: str, loop_vars: Dict, checkpoint_dir: Optional[str] = None):   # ✅ ADDED
    if not loop_vars:
        return
    try:
        _atomic_pickle(_get_path(job_name, ".vars", checkpoint_dir), loop_vars)
    except Exception as e:
        print(f"⚠️  loopz: Could not save loop_vars — {e}")


def load_loop_vars(job_name: str, checkpoint_dir: Optional[str] = None) -> Optional[Dict]:   # ✅ ADDED
    path = _get_path(job_name, ".vars", checkpoint_dir)
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


def clear_loop_vars(job_name: str, checkpoint_dir: Optional[str] = None):   # ✅ ADDED
    p = _get_path(job_name, ".vars", checkpoint_dir)
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# State — ML objects
# ---------------------------------------------------------------------------

def save_state(job_name: str, state: Dict, checkpoint_dir: Optional[str] = None):   # ✅ ADDED
    if not state:
        return
    serialized: Dict[str, Any] = {}
    for key, obj in state.items():
        try:
            serialized[key] = _serialize_obj(obj)
        except Exception as e:
            print(f"⚠️  loopz: Could not save state['{key}'] — {e}")
    serialized["__random_state__"] = save_random_state()
    _atomic_pickle(_get_path(job_name, ".state", checkpoint_dir), serialized)


def load_state(job_name: str, state: Dict, checkpoint_dir: Optional[str] = None) -> bool:   # ✅ ADDED
    path = _get_path(job_name, ".state", checkpoint_dir)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            serialized = pickle.load(f)
    except Exception as e:
        print(f"⚠️  loopz: Could not read state file — {e}")
        return False

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
# Serialization helpers — unchanged from original
# ---------------------------------------------------------------------------

def _serialize_obj(obj: Any) -> Dict:
    try:
        import torch
        import torch.nn as nn

        if isinstance(obj, nn.DataParallel):
            return {
                "__type__": "torch_model_dp",
                "data": {k: v.cpu().clone() for k, v in obj.module.state_dict().items()},
            }

        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            if isinstance(obj, DDP):
                return {
                    "__type__": "torch_model_ddp",
                    "data": {k: v.cpu().clone() for k, v in obj.module.state_dict().items()},
                }
        except (ImportError, Exception):
            pass

        if isinstance(obj, nn.Module):
            return {
                "__type__": "torch_model",
                "data": {k: v.cpu().clone() for k, v in obj.state_dict().items()},
            }

        if isinstance(obj, torch.optim.Optimizer):
            return {"__type__": "torch_optimizer", "data": obj.state_dict()}

        if hasattr(obj, "state_dict") and hasattr(obj, "last_epoch"):
            return {"__type__": "torch_scheduler", "data": obj.state_dict()}

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

        if isinstance(obj, torch.Tensor):
            return {"__type__": "torch_tensor", "data": obj.detach().cpu().clone()}

    except ImportError:
        pass

    if isinstance(obj, np.ndarray):
        return {"__type__": "numpy", "data": obj.copy()}

    if hasattr(obj, "fit") and hasattr(obj, "predict"):
        try:
            return {"__type__": "sklearn", "data": pickle.dumps(obj)}
        except Exception:
            pass

    try:
        return {"__type__": "pickle", "data": pickle.dumps(obj)}
    except Exception as e:
        raise RuntimeError(
            f"loopz: Object of type {type(obj).__name__} is not serializable. "
            f"Original error: {e}"
        )


def _deserialize_into(obj: Any, saved: Dict):
    t = saved.get("__type__")

    if t in ("torch_model", "torch_model_dp", "torch_model_ddp"):
        try:
            import torch
            import torch.nn as nn
            target = obj
            if isinstance(obj, nn.DataParallel):
                target = obj.module
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP
                if isinstance(obj, DDP):
                    target = obj.module
            except (ImportError, Exception):
                pass
            try:
                device = next(target.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            state_dict = {k: v.to(device) for k, v in saved["data"].items()}
            target.load_state_dict(state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"loopz: torch model restore failed — {e}") from e

    elif t == "torch_optimizer":
        try:
            obj.load_state_dict(saved["data"])
        except Exception as e:
            raise RuntimeError(f"loopz: optimizer restore failed — {e}") from e

    elif t == "torch_scheduler":
        try:
            obj.load_state_dict(saved["data"])
        except Exception as e:
            raise RuntimeError(f"loopz: scheduler restore failed — {e}") from e

    elif t == "torch_scaler":
        try:
            obj.load_state_dict(saved["data"])
        except Exception as e:
            raise RuntimeError(f"loopz: GradScaler restore failed — {e}") from e

    elif t == "torch_tensor":
        try:
            import torch
            obj.data = saved["data"].to(obj.device)
        except Exception:
            obj.data = saved["data"]

    elif t == "numpy":
        try:
            obj[:] = saved["data"]
        except Exception as e:
            raise RuntimeError(f"loopz: numpy array restore failed — {e}") from e

    elif t == "sklearn":
        try:
            restored = pickle.loads(saved["data"])
            obj.__dict__.update(restored.__dict__)
        except Exception as e:
            raise RuntimeError(f"loopz: sklearn model restore failed — {e}") from e

    elif t == "pickle":
        try:
            restored = pickle.loads(saved["data"])
            if isinstance(obj, list):
                obj[:] = restored
            elif isinstance(obj, dict):
                obj.clear()
                obj.update(restored)
            elif isinstance(obj, np.ndarray):
                obj[()] = restored
            elif hasattr(obj, "__dict__"):
                obj.__dict__.update(restored.__dict__)
            else:
                print(
                    f"⚠️  loopz: cannot restore primitive type "
                    f"'{type(obj).__name__}' in-place (Python limitation). "
                    f"Wrap it in a list: my_val = [0.0] and access as my_val[0]."
                )
        except Exception as e:
            raise RuntimeError(f"loopz: object restore failed — {e}") from e

    else:
        raise RuntimeError(f"loopz: unknown serialization type '{t}'")