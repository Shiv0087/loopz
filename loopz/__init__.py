"""
loopz — Never lose loop progress again.

Auto-saves and resumes any Python loop from exactly where it crashed.

Works for:
    • File / image / dataset processing
    • ML model training (PyTorch, Sklearn, …)
    • Web scraping & API pagination
    • Large downloads
    • Any long-running Python loop

Quick start::

    import loopz

    @loopz.track("my_job", save_every=100)
    def process(item):
        do_something(item)

    process(my_large_list)
    # crash at 60k?  run again → resumes from 60k ✅

ML training::

    running_loss = [0.0]          # wrap in list so loopz can restore in-place

    @loopz.track(
        "training",
        save_every=1,
        state={"model": model, "optimizer": opt, "scheduler": sched},
        loop_vars={"running_loss": running_loss},
    )
    def train_epoch(epoch):
        loss = run_one_epoch(model, loader, opt, sched)
        running_loss[0] += loss

    train_epoch(range(num_epochs))
"""

from .decorator import track
from .tracker import (
    load_progress,
    clear_progress,
    list_jobs,
    save_progress,
    save_state,
    load_state,
    save_loop_vars,
    load_loop_vars,
    clear_loop_vars,
    save_random_state,
    restore_random_state,
)

__version__ = "1.0.0"
__author__  = "Shivrajsinh Jadeja"
__email__   = "jadejas.k@gmail.com"
__license__ = "MIT"
__url__     = "https://github.com/Shiv0087/loopz"

__all__ = [
    # Core
    "track",
    "status",
    "reset",
    "reset_all",
    # Lower-level helpers (for power users)
    "load_progress",
    "clear_progress",
    "list_jobs",
    "save_state",
    "load_state",
    "save_loop_vars",
    "load_loop_vars",
    "save_random_state",
    "restore_random_state",
]


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def status(checkpoint_dir=None):
    """
    Print a summary of all saved (incomplete) jobs.

    Parameters
    ----------
    checkpoint_dir : str or os.PathLike, optional
        Directory to look for checkpoints in.
        Must match the checkpoint_dir used in @loopz.track().
        Defaults to ~/.loopz/ if not provided.

    Example
    -------
    ::

        loopz.status()
        # or, if you used a custom dir:
        loopz.status(checkpoint_dir="./checkpoints/")
    """
    from pathlib import Path
    # expanduser() so ~/my_dir works correctly
    base_dir = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
    jobs = list_jobs(base_dir=base_dir)
    if not jobs:
        print("\n✅ loopz: No saved jobs — all loops completed cleanly!\n")
        return
    print(f"\n📋 loopz — {len(jobs)} saved job(s):\n")
    for j in jobs:
        print(f"  🔁 {j['job_name']}")
        print(f"     Progress : {j['index']}/{j['total']} ({j['percent']}%)")
        print(f"     Saved at : {j['saved_at']}")
        if j.get("meta", {}).get("error"):
            print(f"     Crashed  : {j['meta']['error']}")
        print()


def reset(job_name: str, checkpoint_dir=None):
    """
    Delete ALL saved data for `job_name` — it will start fresh next run.

    Parameters
    ----------
    job_name : str
        The name passed to ``@loopz.track(job_name=...)``.

    checkpoint_dir : str or os.PathLike, optional
        Directory to look for checkpoints in.
        Must match the checkpoint_dir used in @loopz.track().
        Defaults to ~/.loopz/ if not provided.

    Example
    -------
    ::

        loopz.reset("extract_features")
        # or, if you used a custom dir:
        loopz.reset("extract_features", checkpoint_dir="./checkpoints/")
        # → 🗑️  loopz: 'extract_features' cleared. Will start fresh next run.
    """
    from pathlib import Path
    # expanduser() so ~/my_dir works correctly
    base_dir = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
    clear_progress(job_name, base_dir=base_dir)
    clear_loop_vars(job_name, base_dir=base_dir)
    print(f"🗑️  loopz: '{job_name}' cleared. Will start fresh next run.")


def reset_all(checkpoint_dir=None):
    """
    Delete saved data for EVERY job.

    Use with care — this cannot be undone.

    Parameters
    ----------
    checkpoint_dir : str or os.PathLike, optional
        Directory to look for checkpoints in.
        Must match the checkpoint_dir used in @loopz.track().
        Defaults to ~/.loopz/ if not provided.

    Example
    -------
    ::

        loopz.reset_all()
        # or, if you used a custom dir:
        loopz.reset_all(checkpoint_dir="./checkpoints/")
        # → 🗑️  loopz: All N saved job(s) cleared.
    """
    from pathlib import Path
    # expanduser() so ~/my_dir works correctly
    base_dir = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
    jobs = list_jobs(base_dir=base_dir)
    for j in jobs:
        clear_progress(j["job_name"], base_dir=base_dir)
        clear_loop_vars(j["job_name"], base_dir=base_dir)
    n = len(jobs)
    print(f"🗑️  loopz: All {n} saved job(s) cleared.")