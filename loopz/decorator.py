"""
decorator.py — The @loopz.track decorator.

Key guarantees:
  - Progress is saved at EVERY checkpoint AND at crash time
  - Final checkpoint is written before clear — no gap at the end
  - State, loop_vars, and random seed are all bundled per checkpoint
  - Keyboard interrupt treated like any crash — progress saved
  - Clear warning if save_every is large with heavy state objects
  - Resume prints exactly what was restored so user can verify
"""

import time
import inspect
import asyncio
import concurrent.futures
import functools
from typing import Any, Callable, Dict, Iterable, Optional

from .tracker import (
    save_progress,
    load_progress,
    clear_progress,
    save_state,
    load_state,
    save_loop_vars,
    load_loop_vars,
    clear_loop_vars,
)

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _make_bar(total: int, initial: int, desc: str):
    """Return a tqdm bar, or a no-op context manager if tqdm is absent."""
    if _HAS_TQDM:
        return _tqdm(
            total=total,
            initial=initial,
            desc=desc,
            unit="it",
            dynamic_ncols=True,
            colour="green",
            leave=True,
        )
    # Minimal fallback
    class _NoBar:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): pass
    return _NoBar()


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------

def track(
    job_name:   str,
    save_every: int                     = 10,
    state:      Optional[Dict]          = None,
    loop_vars:  Optional[Dict]          = None,
    notify:     Optional[Callable]      = None,
    checkpoint_dir: Optional[str] = None,
):
    """
    Decorator — auto-saves and resumes any Python loop.

    Parameters
    ----------
    job_name : str
        Unique identifier for this job.  Use a descriptive name so
        ``loopz.status()`` output is readable.

    save_every : int, default 10
        Save a checkpoint every N iterations.
        For heavy model state, use save_every=1.
        Tip: loopz warns you automatically when this may be risky.

    state : dict, optional
        ML objects to persist across crashes.
        Supported types:
            • torch.nn.Module (and DataParallel / DDP)
            • torch.optim.Optimizer  (Adam, SGD, …)
            • torch.optim.lr_scheduler.*
            • torch.cuda.amp.GradScaler
            • torch.Tensor
            • numpy.ndarray
            • sklearn estimators
            • any picklable object
        Example::
            state={"model": model, "optimizer": opt, "scheduler": sched}

    loop_vars : dict, optional
        Plain Python accumulators that live INSIDE the loop.
        Must be wrapped in a list so loopz can mutate them in-place::
            running_loss = [0.0]
            best_acc     = [0.0]
            loop_vars={"running_loss": running_loss, "best_acc": best_acc}

    notify : callable, optional
        Called with a message string on completion OR crash.
        Examples::
            notify=print
            notify=lambda m: requests.post(webhook, json={"text": m})

    Usage
    -----
    Simple file processing::

        @loopz.track("process_images", save_every=100)
        def process(path):
            extract_features(path)

        process(all_image_paths)

    Full ML training loop::

        running_loss = [0.0]

        @loopz.track(
            "training",
            save_every=1,
            state={"model": model, "optimizer": opt, "scheduler": sched},
            loop_vars={"running_loss": running_loss},
        )
        def train(epoch):
            loss = train_one_epoch(model, loader, opt)
            sched.step()
            running_loss[0] += loss

        train(range(num_epochs))
    """

    # Validate save_every
    if not isinstance(save_every, int) or save_every < 1:
        raise ValueError(f"loopz: save_every must be a positive integer, got {save_every!r}")

    def decorator(func: Callable):
        _is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def wrapper(iterable: Iterable, *args, **kwargs):
            items = list(iterable)
            total = len(items)

            # ---- save_every warnings (inside wrapper so we know total) ----
            if state and save_every > 5:
                print(
                    f"⚠️  loopz: save_every={save_every} with state objects — "
                    f"a crash between checkpoints loses up to {save_every} steps of weights. "
                    f"Consider save_every=1 for training loops."
                )
            elif state and save_every == 1 and total > 500:
                print(
                    f"⚠️  loopz: save_every=1 with {total} steps — loopz writes a full "
                    f"checkpoint every single step. For large models this is heavy I/O. "
                    f"If saving is slow, try save_every=5 for good crash coverage with less I/O."
                )

            if total == 0:
                print(f"\n⚠️  loopz: '{job_name}' received empty list — nothing to do.\n")
                return

            # ----------------------------------------------------------------
            # Resume check
            # ----------------------------------------------------------------
            saved       = load_progress(job_name,checkpoint_dir=checkpoint_dir)
            start_index = 0

            if saved and 0 < saved["index"] < total:
                start_index = saved["index"]
                pct         = saved["percent"]
                print(
                    f"\n🔁 loopz: Resuming '{job_name}' "
                    f"from {start_index}/{total} ({pct}% done)"
                )
                print(f"   Last saved  : {saved['saved_at']}")
                if saved.get("meta", {}).get("error"):
                    print(f"   Crashed on  : {saved['meta']['error']}")

                # Restore ML state
                if state:
                    ok = load_state(job_name, state, checkpoint_dir=checkpoint_dir)
                    tag = "✅" if ok else "⚠️  (no saved state found)"
                    print(f"   State       : {list(state.keys())} {tag}")

                # Restore loop vars
                if loop_vars is not None:
                    saved_vars = load_loop_vars(job_name, checkpoint_dir=checkpoint_dir)
                    if saved_vars:
                        for k, v in saved_vars.items():
                            if k in loop_vars:
                                target     = loop_vars[k]
                                saved_val  = v
                                # list wrapper pattern: [value]
                                if isinstance(target, list) and isinstance(saved_val, list):
                                    target[0] = saved_val[0]
                                # loopz.Var
                                elif _is_var(target) and _is_var(saved_val):
                                    target.value = saved_val.value
                                # any other mutable container
                                elif hasattr(target, "__dict__") and hasattr(saved_val, "__dict__"):
                                    target.__dict__.update(saved_val.__dict__)
                        print(f"   Loop vars   : {list(loop_vars.keys())} ✅")
                print()

            else:
                # Fresh start — clear any stale data from a previous full run
                clear_progress(job_name, checkpoint_dir=checkpoint_dir)
                print(f"\n🟢 loopz: Starting '{job_name}' — {total} items")
                if state:
                    print(f"   Tracking    : {list(state.keys())}")
                if loop_vars:
                    print(f"   Loop vars   : {list(loop_vars.keys())}")
                print()

            # ----------------------------------------------------------------
            # Main loop
            # ----------------------------------------------------------------
            start_time = time.time()
            completed  = 0

            def _checkpoint(current_i: int, elapsed: float):
                """Save progress + state + loop_vars.
                ANY I/O error is caught and printed as a warning —
                a disk-full or permissions error must NEVER crash the
                user's loop. The loop keeps running; the next checkpoint
                will try again.
                """
                try:
                    save_progress(
                        job_name, current_i, total,
                        meta={"elapsed_sec": round(elapsed, 1)},
                        checkpoint_dir=checkpoint_dir
                    )
                except Exception as e:
                    print(f"\n⚠️  loopz: could not save progress checkpoint — {e}")
                    return   # skip state/vars save too if progress failed
                try:
                    if state:
                        save_state(job_name, state, checkpoint_dir=checkpoint_dir)
                except Exception as e:
                    print(f"\n⚠️  loopz: could not save state checkpoint — {e}")
                try:
                    if loop_vars is not None:
                        save_loop_vars(job_name, loop_vars, checkpoint_dir=checkpoint_dir)
                except Exception as e:
                    print(f"\n⚠️  loopz: could not save loop_vars checkpoint — {e}")

            try:
                with _make_bar(total, start_index, f"[loopz] {job_name}") as pbar:
                    for i, item in enumerate(
                        items[start_index:], start=start_index
                    ):
                        if _is_async:
                            _run_coro(func(item, *args, **kwargs))
                        else:
                            func(item, *args, **kwargs)
                        completed += 1
                        pbar.update(1)

                        # Checkpoint every N steps
                        if completed % save_every == 0:
                            _checkpoint(i + 1, time.time() - start_time)

                # ---- Successful completion ----
                # Write a final checkpoint covering items after last save_every
                # boundary, then immediately clear.
                elapsed = time.time() - start_time
                _checkpoint(total, elapsed)   # covers the tail gap — BUG FIX
                clear_progress(job_name , checkpoint_dir=checkpoint_dir)
                clear_loop_vars(job_name, checkpoint_dir=checkpoint_dir)

                elapsed_str = _fmt_time(elapsed)
                print(
                    f"\n✅ loopz: '{job_name}' completed! "
                    f"{total} items in {elapsed_str}\n"
                )
                if notify:
                    try:
                        notify(
                            f"✅ loopz '{job_name}' done — "
                            f"{total} items in {elapsed_str}"
                        )
                    except Exception:
                        pass

            except (KeyboardInterrupt, Exception) as exc:
                # ---- Crash / interrupt — save everything ----
                current_i = start_index + completed
                elapsed   = time.time() - start_time
                err_msg   = (
                    "KeyboardInterrupt"
                    if isinstance(exc, KeyboardInterrupt)
                    else str(exc)
                )
                save_progress(
                    job_name, current_i, total,
                    meta={
                        "error":       err_msg,
                        "elapsed_sec": round(elapsed, 1),
                    },
                    checkpoint_dir=checkpoint_dir
                )
                if state:
                    save_state(job_name, state, checkpoint_dir=checkpoint_dir)
                if loop_vars is not None:
                    save_loop_vars(job_name, loop_vars, checkpoint_dir=checkpoint_dir)

                what_saved = "progress"
                if state:
                    what_saved += " + state"
                if loop_vars:
                    what_saved += " + loop_vars"

                print(
                    f"\n💾 loopz: '{job_name}' stopped at "
                    f"{current_i}/{total} ({what_saved} saved). "
                    f"Run again to resume from here.\n"
                )
                if notify:
                    try:
                        notify(
                            f"⚠️  loopz '{job_name}' crashed at "
                            f"{current_i}/{total} — run again to resume."
                        )
                    except Exception:
                        pass
                raise  # re-raise so the user still sees the original error

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------

def _is_var(obj) -> bool:
    """Return True if obj is a loopz.Var instance (avoids circular import)."""
    return type(obj).__name__ == "Var" and hasattr(obj, "value")


def _fmt_time(seconds: float) -> str:
    """Return a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _run_coro(coro):
    """
    Run an async coroutine from sync code — works in both normal scripts
    AND environments with a running event loop (Jupyter, FastAPI, etc.).

    Strategy:
    - No running loop  → asyncio.run()  (standard path)
    - Running loop     → ThreadPoolExecutor with asyncio.run() in a new
                         thread that has its own fresh event loop.
                         No nest_asyncio dependency required.
    """
    try:
        asyncio.get_running_loop()
        # Event loop IS running (Jupyter / FastAPI / etc.)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — normal script / Colab cell
        return asyncio.run(coro)