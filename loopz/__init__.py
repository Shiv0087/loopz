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

def status():
    """
    Print a summary of all saved (i.e. incomplete) jobs.

    Example output::

        📋 loopz — 2 saved job(s):

          🔁 extract_features
             Progress : 60000/118000 (50.85%)
             Saved at : 2026-03-22 14:30:00

          🔁 training
             Progress : 12/50 (24.0%)
             Saved at : 2026-03-22 15:10:00
    """
    jobs = list_jobs()
    if not jobs:
        print("\n✅ loopz: No saved jobs — all loops completed cleanly!\n")
        return

    print(f"\n📋 loopz — {len(jobs)} saved job(s):\n")
    for job in jobs:
        print(f"  🔁 {job['job_name']}")
        print(f"     Progress : {job['index']}/{job['total']} ({job['percent']}%)")
        print(f"     Saved at : {job['saved_at']}")
        if job.get("meta", {}).get("error"):
            print(f"     Crashed  : {job['meta']['error']}")
        elapsed = job.get("meta", {}).get("elapsed_sec")
        if elapsed:
            print(f"     Elapsed  : {elapsed}s before crash")
        print()


def reset(job_name: str):
    """
    Delete ALL saved data for `job_name` — it will start fresh next run.

    Parameters
    ----------
    job_name : str
        The name passed to ``@loopz.track(job_name=...)``.

    Example::

        loopz.reset("extract_features")
        # → 🗑️  loopz: 'extract_features' cleared. Will start fresh next run.
    """
    clear_progress(job_name)
    clear_loop_vars(job_name)
    print(f"🗑️  loopz: '{job_name}' cleared. Will start fresh next run.")


def reset_all():
    """
    Delete saved data for EVERY job.

    Use with care — this cannot be undone.

    Example::

        loopz.reset_all()
        # → 🗑️  loopz: All saved jobs cleared.
    """
    jobs = list_jobs()
    for job in jobs:
        clear_progress(job["job_name"])
        clear_loop_vars(job["job_name"])
    n = len(jobs)
    print(f"🗑️  loopz: All {n} saved job(s) cleared.")  