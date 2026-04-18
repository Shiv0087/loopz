# loopz — Never lose loop progress again.

> Add one decorator. Any Python loop auto-resumes from exactly where it crashed.

[![PyPI version](https://badge.fury.io/py/loopz.svg)](https://badge.fury.io/py/loopz)
[![Python](https://img.shields.io/pypi/pyversions/loopz)](https://pypi.org/project/loopz)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/Shiv0087/loopz/actions/workflows/tests.yml/badge.svg)](https://github.com/Shiv0087/loopz/actions)

---

## The Problem

You are processing 100,000 images. Your Colab session drops at 60,000.  
You are training a model for 50 epochs. Your laptop dies at epoch 30.  
You start over. Every single time.

**loopz fixes this.**

---

## Install

```bash
pip install loopz
```

---

## Quick Start

```python
import loopz

@loopz.track("process_images", save_every=100)
def process(image_path):
    extract_and_save_features(image_path)

process(all_image_paths)   # 💥 crash at 60k?  run again → resumes at 60k ✅
```

That is the entire API for the common case.  
One decorator. One argument. Done.

---

## ML Training — Full State Save

loopz saves and restores your model weights, optimizer state, LR scheduler,
GradScaler, and any accumulators living inside the loop — all automatically.

```python
import loopz
import torch

model     = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

running_loss = [0.0]   # wrap in a list so loopz can restore it in-place
best_acc     = [0.0]

@loopz.track(
    "training",
    save_every  = 1,                                          # save every epoch
    state       = {"model": model, "optimizer": optimizer,
                   "scheduler": scheduler},
    loop_vars   = {"running_loss": running_loss,
                   "best_acc":     best_acc},
    notify      = print,                                      # or send a Telegram/webhook
)
def train(epoch):
    loss, acc = train_one_epoch(model, train_loader, optimizer, scheduler)
    running_loss[0] += loss
    best_acc[0]      = max(best_acc[0], acc)
    print(f"Epoch {epoch} | loss={loss:.4f} | acc={acc:.4f}")

train(range(50))
# 💥 crashes at epoch 12?  run the script again →
# 🔁 loopz: Resuming 'training' from 12/50 (24.0% done)
#    State     : ['model', 'optimizer', 'scheduler'] ✅
#    Loop vars : ['running_loss', 'best_acc'] ✅
```

---

## What Gets Saved

| Object | Supported |
|---|---|
| `torch.nn.Module` | ✅ |
| `torch.nn.DataParallel` | ✅ |
| `torch.nn.parallel.DistributedDataParallel` | ✅ |
| `torch.optim.Optimizer` (Adam, SGD, AdamW, …) | ✅ |
| `torch.optim.lr_scheduler.*` | ✅ |
| `torch.cuda.amp.GradScaler` | ✅ |
| `torch.Tensor` | ✅ |
| `numpy.ndarray` | ✅ |
| sklearn estimator | ✅ |
| Plain Python object (any picklable) | ✅ |
| Python / Numpy / PyTorch / CUDA random state | ✅ |
| Variables inside the loop (`running_loss`, `best_acc`, …) | ✅ |

---

## API Reference

### `@loopz.track(...)`

```python
@loopz.track(
    job_name   = "my_job",    # unique name — used for resume lookup
    save_every = 10,          # checkpoint every N iterations
    state      = {...},       # ML objects to save (optional)
    loop_vars  = {...},       # accumulators inside the loop (optional)
    notify     = callable,    # called on completion or crash (optional)
)
def process(item):
    ...

process(my_list)
```

### `loopz.status()`

Print a summary of all incomplete (saved) jobs.

```
📋 loopz — 1 saved job(s):

  🔁 training
     Progress : 12/50 (24.0%)
     Saved at : 2026-03-22 14:30:00
     Crashed  : training crash at epoch 12
```

### `loopz.reset("job_name")`

Delete all saved data for a job — it will start fresh next run.

### `loopz.reset_all()`

Delete all saved data for every job.

---

## How It Works

1. On every `save_every`-th iteration loopz atomically writes:
   - your loop position (JSON)
   - your ML object weights (`.state`)
   - your loop variables (`.vars`)
   - the full random seed state (Python + Numpy + PyTorch + CUDA)

2. On crash or KeyboardInterrupt, it saves one final checkpoint then re-raises the original exception so your stack trace is still visible.

3. On the next run, loopz detects the saved position, restores all state, and resumes the loop from exactly that index.

4. On clean completion, all saved files are deleted automatically.

---

## Limitations (be honest)

- **Primitives as loop_vars** — `int`, `float`, `str` cannot be mutated in-place in Python. Wrap them in a list: `loss = [0.0]` not `loss = 0.0`.
- **Distributed training (multi-node)** — DDP on a single machine is supported; multi-node DDP across separate machines is not.
- **Custom C++ extensions** — if your model uses custom CUDA ops with non-standard state, manual checkpointing is needed alongside loopz.
- **Non-picklable objects** — if an object in `state=` cannot be pickled, loopz will print a warning and skip it.

---

## License

MIT © Shivrajsinh Jadeja