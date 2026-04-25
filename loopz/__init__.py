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
    from pathlib import Path
    base_dir = Path(checkpoint_dir) if checkpoint_dir else None
    jobs = list_jobs(base_dir=base_dir)
    if not jobs:
        print("\n📋 loopz — no saved jobs.\n")
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
    from pathlib import Path
    base_dir = Path(checkpoint_dir) if checkpoint_dir else None
    clear_progress(job_name, base_dir=base_dir)
    clear_loop_vars(job_name, base_dir=base_dir)
    print(f"🗑️  loopz: '{job_name}' reset.")


def reset_all(checkpoint_dir=None):
    from pathlib import Path
    base_dir = Path(checkpoint_dir) if checkpoint_dir else None
    for j in list_jobs(base_dir=base_dir):
        clear_progress(j["job_name"], base_dir=base_dir)
        clear_loop_vars(j["job_name"], base_dir=base_dir)
    print("🗑️  loopz: all jobs reset.")