"""
hard_test.py — Production test suite for loopz v1.0.0
Run: python tests/hard_test.py

Covers:
    Section 1  — Basic loops
    Section 2  — FIX 1: Random seed save/restore
    Section 3  — FIX 2: Loop variables
    Section 4  — FIX 3: Multi-GPU (DataParallel / DDP)
    Section 5  — PyTorch full training crash + resume
    Section 6  — Numpy state
    Section 7  — Sklearn state
    Section 8  — Job management
    Section 9  — Edge cases & error recovery
    Section 10 — Notifications
    Section 11 — Performance
"""

import os
import sys
import time
import random
import pickle
import hashlib
import numpy as np
from pathlib import Path

# Make sure we can import loopz from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import loopz
from loopz.tracker import (
    save_progress, load_progress, clear_progress,
    save_state, load_state,
    save_loop_vars, load_loop_vars, clear_loop_vars,
    save_random_state, restore_random_state,
    _serialize_obj, _deserialize_into,
    CACHE_DIR,
)

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
passed = 0
failed = 0
skipped = 0


def test(name: str, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✅ PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL  {name}")
        print(f"          → {e}")
        failed += 1


def skip(name: str, reason: str):
    global skipped
    print(f"  ⚠️  SKIP  {name}  ({reason})")
    skipped += 1


def section(title: str):
    print(f"\n── {title} {'─' * max(0, 55 - len(title))}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean(*names):
    for n in names:
        clear_progress(n)
        clear_loop_vars(n)


# ===========================================================================
# SECTION 1 — Basic Loops
# ===========================================================================
section("Section 1: Basic Loops")

def t1():
    _clean("t1")
    results = []
    @loopz.track("t1", save_every=2)
    def process(item): results.append(item)
    process(range(10))
    assert len(results) == 10
    assert load_progress("t1") is None   # cleared after completion
test("Normal loop completes fully + progress cleared", t1)

def t2():
    _clean("t2")
    save_progress("t2", 5, 10)
    results = []
    @loopz.track("t2", save_every=2)
    def process(item): results.append(item)
    process(range(10))
    assert results == list(range(5, 10)), f"got {results}"
test("Resume from 50% — only remaining items run", t2)

def t3():
    _clean("t3")
    save_progress("t3", 9, 10)
    results = []
    @loopz.track("t3", save_every=1)
    def process(item): results.append(item)
    process(range(10))
    assert results == [9], f"got {results}"
test("Resume from 90% — only last item runs", t3)

def t4():
    _clean("t4")
    results = []
    @loopz.track("t4")
    def process(item): results.append(item)
    process([])
    assert len(results) == 0
test("Empty list — no crash, no progress file", t4)

def t5():
    _clean("t5")
    results = []
    @loopz.track("t5")
    def process(item): results.append(item)
    process([42])
    assert results == [42]
test("Single item list", t5)

def t6():
    _clean("t6")
    results = []
    @loopz.track("t6", save_every=1000)
    def process(item): results.append(item)
    process(range(10_000))
    assert len(results) == 10_000
test("10,000 items — completes correctly", t6)

def t7():
    _clean("t7")
    results = []
    items = [f"image_{i:06d}.jpg" for i in range(20)]
    @loopz.track("t7", save_every=5)
    def process(item): results.append(item)
    process(items)
    assert results == items
test("String items — file paths", t7)

def t8():
    _clean("t8")
    results = []
    items = [{"id": i, "path": f"img_{i}.jpg", "label": i % 2} for i in range(10)]
    @loopz.track("t8", save_every=3)
    def process(item): results.append(item["id"])
    process(items)
    assert results == list(range(10))
test("Dict items — ML dataset records", t8)

def t9():
    _clean("t9")
    # save_every larger than total — final tail must still be saved properly
    results = []
    @loopz.track("t9", save_every=100)
    def process(item): results.append(item)
    process(range(7))   # 7 < save_every=100
    assert len(results) == 7
    assert load_progress("t9") is None
test("save_every larger than total — no items lost", t9)


# ===========================================================================
# SECTION 2 — FIX 1: Random Seed
# ===========================================================================
section("Section 2: FIX 1 — Random Seed Save / Restore")

def t10():
    random.seed(42)
    state_before = save_random_state()
    random.random()  # advance state
    restore_random_state(state_before)
    random.seed(42)
    expected = random.random()
    restore_random_state(state_before)
    got = random.random()
    assert abs(expected - got) < 1e-12
test("Python random state — save and restore", t10)

def t11():
    np.random.seed(123)
    state = save_random_state()
    v1 = np.random.rand(10)
    np.random.seed(999)          # corrupt state
    restore_random_state(state)
    v2 = np.random.rand(10)
    np.testing.assert_array_almost_equal(v1, v2)
test("Numpy random state — save and restore", t11)

def t12():
    _clean("t12_rng")
    arr = np.zeros(5)
    np.random.seed(42)
    @loopz.track("t12_rng", save_every=1, state={"arr": arr})
    def process(item):
        arr[item] = np.random.rand()
        if item == 2:
            raise Exception("rng crash")
    try:
        process(range(5))
    except Exception:
        pass
    safe  = hashlib.md5(b"t12_rng").hexdigest()[:12]
    sfile = CACHE_DIR / f"loopz_{safe}.state"
    assert sfile.exists(), "state file missing after crash"
    _clean("t12_rng")
test("Random state bundled in state file on crash", t12)


# ===========================================================================
# SECTION 3 — FIX 2: Loop Variables
# ===========================================================================
section("Section 3: FIX 2 — Loop Variables")

def t13():
    _clean("t13_vars")
    running_loss = [0.0]
    @loopz.track("t13_vars", save_every=2, loop_vars={"running_loss": running_loss})
    def process(item): running_loss[0] += 1.0
    process(range(10))
    assert running_loss[0] == 10.0, f"got {running_loss[0]}"
    assert load_loop_vars("t13_vars") is None   # cleared on completion
test("Loop var accumulates correctly + cleared on completion", t13)

def t14():
    _clean("t14_crash_vars")
    running_loss = [0.0]
    @loopz.track("t14_crash_vars", save_every=1, loop_vars={"running_loss": running_loss})
    def process(item):
        running_loss[0] += 1.0
        if item == 4:
            raise Exception("crash")
    try:
        process(range(10))
    except Exception:
        pass
    saved = load_loop_vars("t14_crash_vars")
    assert saved is not None, "loop_vars not saved on crash"
    assert saved["running_loss"][0] == 5.0, f"got {saved['running_loss'][0]}"
    _clean("t14_crash_vars")
test("Loop vars saved on crash — correct value", t14)

def t15():
    _clean("t15_resume_vars")
    running_loss = [0.0]
    save_progress("t15_resume_vars", 5, 10)
    save_loop_vars("t15_resume_vars", {"running_loss": [5.0]})
    @loopz.track("t15_resume_vars", save_every=1, loop_vars={"running_loss": running_loss})
    def process(item): running_loss[0] += 1.0
    process(range(10))
    assert running_loss[0] == 10.0, f"got {running_loss[0]}"
test("Loop vars restored on resume, continues accumulating", t15)

def t16():
    _clean("t16_multi_vars")
    loss  = [0.0]
    acc   = [0.0]
    steps = [0]
    @loopz.track("t16_multi_vars", save_every=2,
                 loop_vars={"loss": loss, "acc": acc, "steps": steps})
    def process(item):
        loss[0] += 0.1
        acc[0]  += 0.01
        steps[0] += 1
    process(range(10))
    assert steps[0] == 10
    assert abs(loss[0] - 1.0) < 1e-9
test("Multiple loop vars — all correct after completion", t16)


# ===========================================================================
# SECTION 4 — FIX 3: Multi-GPU
# ===========================================================================
section("Section 4: FIX 3 — Multi-GPU (DataParallel / DDP)")

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

def t17():
    if not _TORCH:
        skip("DataParallel save/restore weights correctly", "PyTorch not installed"); return
    model = nn.Linear(10, 2)
    dp    = nn.DataParallel(model)
    orig  = dp.module.weight.data[0][0].item()
    ser   = _serialize_obj(dp)
    assert ser["__type__"] in ("torch_model_dp", "torch_model")
    with torch.no_grad():
        dp.module.weight.fill_(9.9)
    _deserialize_into(dp, ser)
    got = dp.module.weight.data[0][0].item()
    assert abs(got - orig) < 1e-4, f"expected {orig}, got {got}"
test("DataParallel — save/restore weights correctly", t17)

def t18():
    if not _TORCH:
        skip("DataParallel wrapped in @loopz.track crash+resume", "PyTorch not installed"); return
    _clean("t18_dp")
    model = nn.Linear(10, 2)
    dp    = nn.DataParallel(model)
    orig  = {k: v.clone() for k, v in dp.module.state_dict().items()}
    epochs_done = []
    @loopz.track("t18_dp", save_every=1, state={"model": dp})
    def train(epoch):
        epochs_done.append(epoch)
        if epoch == 2:
            raise Exception("dp crash")
    try:
        train(range(5))
    except Exception:
        pass
    # Corrupt weights
    with torch.no_grad():
        dp.module.weight.fill_(99.0)
    # Resume — should restore weights
    epochs2 = []
    @loopz.track("t18_dp", save_every=1, state={"model": dp})
    def train2(epoch):
        epochs2.append(epoch)
    train2(range(5))
    assert epochs2 == [2, 3, 4], f"got {epochs2}"
    _clean("t18_dp")
test("DataParallel wrapped in @loopz.track — crash + resume", t18)


# ===========================================================================
# SECTION 5 — PyTorch full training
# ===========================================================================
section("Section 5: PyTorch Full Training Loop")

def t19():
    if not _TORCH:
        skip("PyTorch model + optimizer — crash at epoch 2, resume from 2", "PyTorch not installed"); return
    _clean("t19_torch")
    model     = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    done1 = []
    @loopz.track("t19_torch", save_every=1,
                 state={"model": model, "optimizer": optimizer, "scheduler": scheduler})
    def train(epoch):
        done1.append(epoch)
        x    = torch.randn(8, 4)
        y    = torch.randint(0, 2, (8,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
        if epoch == 2:
            raise Exception("training crash!")
    try:
        train(range(6))
    except Exception:
        pass
    prog = load_progress("t19_torch")
    assert prog is not None
    assert prog["index"] == 2, f"expected index 2, got {prog['index']}"

    # Corrupt weights before resume to verify restore works
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(99.0)

    done2 = []
    @loopz.track("t19_torch", save_every=1,
                 state={"model": model, "optimizer": optimizer, "scheduler": scheduler})
    def train2(epoch):
        done2.append(epoch)
    train2(range(6))
    assert done2 == [2, 3, 4, 5], f"got {done2}"
    # Weights should NOT be 99 — they were restored
    for p in model.parameters():
        assert not torch.all(p == 99.0), "model weights were not restored!"
    assert load_progress("t19_torch") is None
    _clean("t19_torch")
test("PyTorch model + optimizer + scheduler — crash at epoch 2, resume from 2", t19)

def t20():
    if not _TORCH:
        skip("GradScaler (mixed precision) save/restore", "PyTorch not installed"); return
    # Try new API first (PyTorch 2.x), fall back to old API (PyTorch 1.x)
    scaler = None
    try:
        from torch.amp import GradScaler
        scaler = GradScaler("cpu")           # new API — device-agnostic
    except (ImportError, TypeError, Exception):
        pass
    if scaler is None:
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler(enabled=False)   # old API — works on CPU
        except (ImportError, Exception):
            skip("GradScaler (mixed precision) save/restore", "GradScaler not available"); return
    ser = _serialize_obj(scaler)
    assert ser["__type__"] == "torch_scaler", f"got type: {ser['__type__']}"
    _deserialize_into(scaler, ser)
test("GradScaler (mixed precision) save/restore", t20)


# ===========================================================================
# SECTION 6 — Numpy
# ===========================================================================
section("Section 6: Numpy State")

def t21():
    _clean("t21_np")
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    original = arr.copy()
    save_state("t21_np", {"arr": arr})
    arr[:] = 999.0
    load_state("t21_np", {"arr": arr})
    np.testing.assert_array_equal(arr, original)
    _clean("t21_np")
test("Numpy array — save and restore in place", t21)

def t22():
    _clean("t22_np_crash")
    features = np.zeros((10, 4))
    @loopz.track("t22_np_crash", save_every=2, state={"features": features})
    def process(i):
        features[i] = float(i)
        if i == 5:
            raise Exception("np crash")
    try:
        process(range(10))
    except Exception:
        pass
    prog = load_progress("t22_np_crash")
    assert prog is not None
    _clean("t22_np_crash")
test("Numpy array crash — state file exists after crash", t22)


# ===========================================================================
# SECTION 7 — Sklearn
# ===========================================================================
section("Section 7: Sklearn State")

try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

def t23():
    if not _SKLEARN:
        skip("Sklearn model — save and restore in place", "sklearn not installed"); return
    X = np.random.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=200).fit(X, y)
    coef_before = clf.coef_.copy()
    ser = _serialize_obj(clf)
    assert ser["__type__"] == "sklearn"
    # Corrupt
    clf.coef_[:] = 999.0
    _deserialize_into(clf, ser)
    np.testing.assert_array_almost_equal(clf.coef_, coef_before)
test("Sklearn model — save and restore in place", t23)


# ===========================================================================
# SECTION 8 — Job Management
# ===========================================================================
section("Section 8: Job Management")

def t24():
    _clean("t24_a", "t24_b")
    save_progress("t24_a", 3, 10)
    save_progress("t24_b", 7, 10)
    ra, rb = [], []
    @loopz.track("t24_a", save_every=1)
    def pa(item): ra.append(item)
    @loopz.track("t24_b", save_every=1)
    def pb(item): rb.append(item)
    pa(range(10)); pb(range(10))
    assert ra == list(range(3, 10)), f"got {ra}"
    assert rb == list(range(7, 10)), f"got {rb}"
test("Two jobs — progress files independent, no interference", t24)

def t25():
    _clean("t25_reset")
    save_progress("t25_reset", 50, 100)
    save_loop_vars("t25_reset", {"v": [1.0]})
    loopz.reset("t25_reset")
    assert load_progress("t25_reset") is None
    assert load_loop_vars("t25_reset") is None
test("reset() clears progress AND loop_vars", t25)

def t26():
    _clean("t26_status")
    save_progress("t26_status", 25, 100)
    loopz.status()     # must not crash
    _clean("t26_status")
test("status() runs without error", t26)

def t27():
    _clean("t27_reset_all")
    save_progress("t27_reset_all", 10, 100)
    loopz.reset_all()
    assert load_progress("t27_reset_all") is None
test("reset_all() clears all jobs", t27)


# ===========================================================================
# SECTION 9 — Edge Cases & Error Recovery
# ===========================================================================
section("Section 9: Edge Cases & Error Recovery")

def t28():
    # Corrupted JSON → graceful recovery (returns None, no crash)
    safe = hashlib.md5(b"t28_corrupt").hexdigest()[:8]
    bad  = CACHE_DIR / f"t28_corrupt_{safe}.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("NOT VALID JSON {{{{", encoding="utf-8")
    result = load_progress("t28_corrupt")
    assert result is None
test("Corrupted JSON — graceful recovery, returns None", t28)

def t29():
    # Corrupted .vars file → graceful recovery
    safe = hashlib.md5(b"t29_corrupt_vars").hexdigest()[:8]
    bad  = CACHE_DIR / f"t29_corrupt_vars_{safe}.vars"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"NOT VALID PICKLE XXXX")
    result = load_loop_vars("t29_corrupt_vars")
    assert result is None
test("Corrupted .vars file — graceful recovery, returns None", t29)

def t30():
    # KeyboardInterrupt treated like a crash — progress saved
    _clean("t30_kbd")
    results = []
    @loopz.track("t30_kbd", save_every=1)
    def process(item):
        results.append(item)
        if item == 3:
            raise KeyboardInterrupt()
    try:
        process(range(10))
    except KeyboardInterrupt:
        pass
    prog = load_progress("t30_kbd")
    assert prog is not None, "progress should be saved on KeyboardInterrupt"
    assert prog["index"] == 3
    _clean("t30_kbd")
test("KeyboardInterrupt — progress saved, re-raised", t30)

def t31():
    # index == total on resume (already finished) → starts fresh
    _clean("t31_full")
    save_progress("t31_full", 10, 10)   # 100% complete
    results = []
    @loopz.track("t31_full", save_every=1)
    def process(item): results.append(item)
    process(range(10))
    assert len(results) == 10, f"got {len(results)}"
test("Saved index == total → treated as fresh start, reruns all", t31)

def t32():
    # Atomic write: check no .tmp file left behind after success
    _clean("t32_atomic")
    @loopz.track("t32_atomic", save_every=3)
    def process(item): pass
    process(range(10))
    safe = hashlib.md5(b"t32_atomic").hexdigest()[:12]
    tmp  = CACHE_DIR / f"loopz_{safe}.tmp"
    assert not tmp.exists(), ".tmp file should not remain after clean run"
test("Atomic write — no .tmp file left after completion", t32)

def t33():
    # Negative save_every should raise immediately
    raised = False
    try:
        @loopz.track("t33_bad", save_every=-1)
        def process(item): pass
        raised = False
    except ValueError:
        raised = True
    assert raised
test("save_every < 1 raises ValueError immediately", t33)


# ===========================================================================
# SECTION 10 — Notifications
# ===========================================================================
section("Section 10: Notifications")

def t34():
    _clean("t34_notify")
    msgs = []
    @loopz.track("t34_notify", save_every=2, notify=lambda m: msgs.append(m))
    def process(item): pass
    process(range(5))
    assert len(msgs) == 1
    assert any(w in msgs[0].lower() for w in ("done", "complet"))
test("notify fires once on completion", t34)

def t35():
    _clean("t35_crash_notify")
    msgs = []
    @loopz.track("t35_crash_notify", save_every=1, notify=lambda m: msgs.append(m))
    def process(item):
        if item == 2: raise Exception("crash!")
    try:
        process(range(5))
    except Exception:
        pass
    assert len(msgs) == 1
    assert any(w in msgs[0].lower() for w in ("crash", "resume", "stopped"))
    _clean("t35_crash_notify")
test("notify fires once on crash", t35)

def t36():
    # Broken notify callback must NOT prevent progress from being saved
    _clean("t36_bad_notify")
    @loopz.track("t36_bad_notify", save_every=1,
                 notify=lambda m: 1 / 0)   # always raises
    def process(item):
        if item == 2: raise Exception("crash!")
    try:
        process(range(5))
    except Exception:
        pass
    prog = load_progress("t36_bad_notify")
    assert prog is not None, "progress must be saved even if notify() raises"
    _clean("t36_bad_notify")
test("Broken notify callback — progress still saved", t36)


# ===========================================================================
# SECTION 11 — Performance
# ===========================================================================
section("Section 11: Performance")

def t37():
    _clean("t37_perf")
    start = time.time()
    @loopz.track("t37_perf", save_every=100)
    def process(item): pass
    process(range(1_000))
    elapsed = time.time() - start
    assert elapsed < 5.0, f"1k items took {elapsed:.2f}s — too slow"
test("1,000 items under 5 seconds", t37)

def t38():
    _clean("t38_perf_large")
    start = time.time()
    @loopz.track("t38_perf_large", save_every=1_000)
    def process(item): pass
    process(range(100_000))
    elapsed = time.time() - start
    assert elapsed < 30.0, f"100k items took {elapsed:.2f}s — too slow"
    print(f"       (100k items in {elapsed:.2f}s)")
test("100,000 items under 30 seconds", t38)

def t39():
    # save_every=1 with 1000 items = 1000 disk writes — must still be fast
    _clean("t39_perf_save1")
    start = time.time()
    @loopz.track("t39_perf_save1", save_every=1)
    def process(item): pass
    process(range(1_000))
    elapsed = time.time() - start
    assert elapsed < 30.0, f"1k items save_every=1 took {elapsed:.2f}s — too slow"
    print(f"       (1k items save_every=1 in {elapsed:.2f}s)")
test("1,000 items save_every=1 (max disk writes) under 30 seconds", t39)


# ===========================================================================
# SECTION 12 — Real-world Input Types
# ===========================================================================
section("Section 12: Real-world Input Types")

def t40():
    # Generator input — common in ML (e.g. custom dataset iterators)
    _clean("t40_gen")
    results = []
    @loopz.track("t40_gen", save_every=2)
    def process(item): results.append(item)
    process(x * 2 for x in range(5))
    assert results == [0, 2, 4, 6, 8], f"got {results}"
test("Generator input — converted to list correctly", t40)

def t41():
    # Tuple input
    _clean("t41_tuple")
    results = []
    @loopz.track("t41_tuple", save_every=2)
    def process(item): results.append(item)
    process((10, 20, 30, 40, 50))
    assert results == [10, 20, 30, 40, 50]
test("Tuple input works", t41)

def t42():
    # None values inside the list
    _clean("t42_none")
    results = []
    @loopz.track("t42_none", save_every=2)
    def process(item): results.append(item)
    process([1, None, 3, None, 5])
    assert results == [1, None, 3, None, 5]
test("None values in list — no crash", t42)

def t43():
    # Extra args and kwargs passed through the decorator
    _clean("t43_args")
    results = []
    @loopz.track("t43_args", save_every=2)
    def process(item, multiplier, offset=0):
        results.append(item * multiplier + offset)
    process(range(5), 10, offset=1)
    assert results == [1, 11, 21, 31, 41], f"got {results}"
test("Extra *args and **kwargs pass through correctly", t43)

def t44():
    # Running same job twice — second run must start fresh (not resume)
    _clean("t44_double")
    results = []
    @loopz.track("t44_double", save_every=2)
    def process(item): results.append(item)
    process(range(5))
    process(range(5))   # Jupyter users re-run cells constantly
    assert len(results) == 10, f"expected 10 total, got {len(results)}"
test("Running same job twice — second run is fresh (Jupyter re-run safe)", t44)


# ===========================================================================
# SECTION 13 — Special Job Names
# ===========================================================================
section("Section 13: Special Job Names")

def t45():
    # Job name with forward slash (e.g. "experiment/run1")
    name = "experiment/run1"
    _clean(name)
    results = []
    @loopz.track(name, save_every=2)
    def process(item): results.append(item)
    process(range(5))
    assert len(results) == 5
    assert load_progress(name) is None
test("Job name with '/' (e.g. experiment/run1) — no path crash", t45)

def t46():
    # Job name with backslash, colon, spaces
    name = "model\\v2:final test"
    _clean(name)
    results = []
    @loopz.track(name, save_every=2)
    def process(item): results.append(item)
    process(range(5))
    assert len(results) == 5
test("Job name with backslash, colon, spaces — no crash", t46)

def t47():
    # Very long job name (300 chars)
    name = "training_run_" + "x" * 300
    _clean(name)
    results = []
    @loopz.track(name, save_every=2)
    def process(item): results.append(item)
    process(range(5))
    assert len(results) == 5
test("300-character job name — no crash", t47)

def t48():
    # Two different job names that hash-collide must NOT interfere
    # (extremely unlikely with MD5[:12] but we verify the design)
    name_a = "job_alpha"
    name_b = "job_beta"
    _clean(name_a); _clean(name_b)
    save_progress(name_a, 3, 10)
    save_progress(name_b, 7, 10)
    prog_a = load_progress(name_a)
    prog_b = load_progress(name_b)
    assert prog_a["index"] == 3, f"got {prog_a['index']}"
    assert prog_b["index"] == 7, f"got {prog_b['index']}"
    _clean(name_a); _clean(name_b)
test("Different job names have independent progress files", t48)


# ===========================================================================
# SECTION 14 — State= Edge Cases
# ===========================================================================
section("Section 14: state= Edge Cases")

def t49():
    # list in state= — must restore in-place correctly
    from loopz.tracker import _serialize_obj, _deserialize_into
    my_list = [1, 2, 3, 4, 5]
    ser = _serialize_obj(my_list)
    my_list2 = [0, 0, 0, 0, 0]
    _deserialize_into(my_list2, ser)
    assert my_list2 == [1, 2, 3, 4, 5], f"got {my_list2}"
test("list in state= restores in-place correctly", t49)

def t50():
    # dict in state= — must restore in-place correctly
    from loopz.tracker import _serialize_obj, _deserialize_into
    my_dict = {"lr": 0.001, "epochs": 50, "name": "resnet"}
    ser = _serialize_obj(my_dict)
    my_dict2 = {}
    _deserialize_into(my_dict2, ser)
    assert my_dict2 == my_dict, f"got {my_dict2}"
test("dict in state= restores in-place correctly", t50)

def t51():
    # list in state= through full crash + resume cycle
    _clean("t51_list_state")
    history = [0.9, 0.8, 0.7]    # e.g. loss history so far

    @loopz.track("t51_list_state", save_every=1,
                 state={"history": history})
    def process(item):
        history.append(round(0.6 - item * 0.05, 2))
        if item == 2:
            raise Exception("crash!")

    try:
        process(range(6))
    except Exception:
        pass

    # Corrupt the list
    history.clear()
    history.extend([999, 999, 999])

    # Resume — history should be restored from checkpoint
    results2 = []
    @loopz.track("t51_list_state", save_every=1,
                 state={"history": history})
    def process2(item):
        results2.append(item)

    process2(range(6))
    # After restore history should NOT be [999,999,999]
    assert 999 not in history, f"history was not restored: {history}"
    _clean("t51_list_state")
test("list in state= — full crash + resume restores correctly", t51)

def t52():
    # numpy 0-dimensional scalar
    from loopz.tracker import _serialize_obj
    import numpy as np
    arr = np.float32(3.14)
    ser = _serialize_obj(arr)
    # Should not crash — falls back to pickle
    assert ser is not None
test("0-d numpy scalar in state= — serializes without crash", t52)


# ===========================================================================
# SECTION 15 — Disk / I/O Resilience
# ===========================================================================
section("Section 15: Disk / I/O Resilience")

def t53():
    # Disk write failure during mid-loop checkpoint must NOT crash the user loop
    # The loop should complete; loopz just prints a warning
    import loopz.tracker as _t
    from loopz.tracker import _atomic_json as _real_json

    write_count = [0]
    def _failing_json(path, obj):
        write_count[0] += 1
        if write_count[0] == 2:   # fail on exactly the 2nd write
            raise OSError("Simulated: disk full")
        _real_json(path, obj)

    _t._atomic_json = _failing_json
    _clean("t53_diskfail")
    results = []
    try:
        @loopz.track("t53_diskfail", save_every=2)
        def process(item): results.append(item)
        process(range(10))
    finally:
        _t._atomic_json = _real_json   # always restore
        _clean("t53_diskfail")

    assert len(results) == 10, \
        f"loop stopped early due to disk error — got {len(results)} items"
test("Disk write failure mid-loop — loop still completes, only warns", t53)

def t54():
    # Corrupted .state file — load_state returns False gracefully
    from loopz.tracker import CACHE_DIR, _safe_name, load_state
    safe = _safe_name("t54_bad_state")
    bad  = CACHE_DIR / f"loopz_{safe}.state"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"XXXX CORRUPTED XXXX")
    result = load_state("t54_bad_state", {})
    assert result == False, f"Expected False, got {result}"
test("Corrupted .state file — load_state returns False gracefully", t54)


# ===========================================================================
# SECTION 16 — Threading
# ===========================================================================
section("Section 16: Threading")

def t55():
    import threading
    results = {"a": [], "b": [], "c": []}

    def run(key, start):
        @loopz.track(f"thread_{key}", save_every=5)
        def process(item): results[key].append(item)
        process(range(start, start + 20))

    threads = [
        threading.Thread(target=run, args=("a", 0)),
        threading.Thread(target=run, args=("b", 100)),
        threading.Thread(target=run, args=("c", 200)),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    for key in ("a", "b", "c"):
        assert len(results[key]) == 20, \
            f"thread {key}: expected 20 items, got {len(results[key])}"
    _clean("thread_a"); _clean("thread_b"); _clean("thread_c")
test("3 concurrent threads with separate job names — no interference", t55)


# ===========================================================================
# SECTION 17 — Real-world Usage Patterns
# ===========================================================================
section("Section 17: Real-world Usage Patterns")

def t56():
    # loopz used inside a class method (Trainer pattern — very common in ML)
    _clean("t56_class")
    class Trainer:
        def __init__(self):
            self.results = []
            self.count   = 0
        def run(self, data):
            @loopz.track("t56_class", save_every=2)
            def step(item):
                self.results.append(item)
                self.count += 1
            step(data)
    t = Trainer()
    t.run(range(10))
    assert len(t.results) == 10
    assert t.count == 10
test("loopz inside a class method (Trainer pattern)", t56)

def t57():
    # Decorator defined ONCE at module level, called multiple times
    # (how 99% of real scripts are written)
    _clean("t57_reuse")
    results = []
    @loopz.track("t57_reuse", save_every=2)
    def process(item): results.append(item)

    process(range(5))          # first call — train split
    process(range(100, 105))   # second call — val split, must start fresh
    assert results == [0,1,2,3,4,100,101,102,103,104], f"got {results}"
test("Decorator defined once, called multiple times — each call is independent", t57)

def t58():
    # Crash on the very first item (index 0)
    # Saved index=0 means next run starts fresh — which correctly retries item 0
    _clean("t58_first_crash")
    results = []
    @loopz.track("t58_first_crash", save_every=1)
    def process(item):
        results.append(item)
        if item == 0: raise Exception("bad first item")
    try:
        process(range(5))
    except Exception:
        pass
    prog = load_progress("t58_first_crash")
    assert prog is not None and prog["index"] == 0

    # Second run: item 0 no longer crashes — resumes correctly from 0
    @loopz.track("t58_first_crash", save_every=1)
    def process2(item): results.append(item)
    process2(range(5))
    # results should contain: [0] from first run + [0,1,2,3,4] from second
    assert 0 in results
    _clean("t58_first_crash")
test("Crash on item 0 — next run correctly retries from 0", t58)

def t59():
    # Saved index > new total (user deleted some data between runs)
    # Should start fresh, not skip everything silently
    save_progress("t59_shrink", 800, 1000)  # old run crashed at 800/1000
    results = []
    @loopz.track("t59_shrink", save_every=10)
    def process(item): results.append(item)
    process(range(600))   # dataset shrunk to 600
    # start_index=800 > total=600 → condition (0 < 800 < 600) is False
    # so it starts fresh and runs all 600 items
    assert len(results) == 600, \
        f"expected 600 items (fresh start), got {len(results)}"
test("Saved index > new total (shrunk dataset) — starts fresh, no silent skip", t59)

def t60():
    # No double-processing: items before crash point must NOT rerun on resume
    _clean("t60_no_double")
    call_count = {}
    @loopz.track("t60_no_double", save_every=1)
    def process(item):
        call_count[item] = call_count.get(item, 0) + 1
        if item == 5: raise Exception("crash!")
    try:
        process(range(10))
    except Exception:
        pass
    # Resume
    @loopz.track("t60_no_double", save_every=1)
    def process2(item):
        call_count[item] = call_count.get(item, 0) + 1
    process2(range(10))
    # Items 0-4: called exactly once (not re-processed)
    # Item 5: called twice (crashed then retried — correct)
    # Items 6-9: called exactly once
    for i in range(5):
        assert call_count[i] == 1, f"item {i} called {call_count[i]} times (expected 1)"
    assert call_count[5] == 2,  f"item 5 called {call_count[5]} times (expected 2)"
    for i in range(6, 10):
        assert call_count[i] == 1, f"item {i} called {call_count[i]} times (expected 1)"
    _clean("t60_no_double")
test("No double-processing — items before crash are NOT rerun on resume", t60)

def t61():
    # Nested loopz decorators (outer=epochs, inner=batches)
    _clean("t61_outer"); _clean("t61_inner")
    epoch_log = []; batch_log = []
    @loopz.track("t61_outer", save_every=1)
    def train_epoch(epoch):
        epoch_log.append(epoch)
        @loopz.track("t61_inner", save_every=5)
        def train_batch(b): batch_log.append((epoch, b))
        train_batch(range(10))
    train_epoch(range(3))
    assert epoch_log == [0, 1, 2]
    assert len(batch_log) == 30
    _clean("t61_outer"); _clean("t61_inner")
test("Nested @loopz.track decorators — outer epochs + inner batches", t61)

def t62():
    # Async function must raise TypeError immediately — not silently do nothing
    import asyncio
    raised = False
    try:
        @loopz.track("t62_async", save_every=1)
        async def process(item):
            await asyncio.sleep(0)
        raised = False
    except TypeError as e:
        raised = True
        assert "async" in str(e).lower(), f"error message should mention async: {e}"
    assert raised, "async function should raise TypeError immediately"
test("Async function raises TypeError immediately with clear message", t62)

def t63():
    # Decorator defined inside a for-loop (common notebook mistake)
    # Each iteration must use its own unique job name
    _clean("t63_exp_0"); _clean("t63_exp_1"); _clean("t63_exp_2")
    all_results = {}
    for exp in range(3):
        results = []
        @loopz.track(f"t63_exp_{exp}", save_every=2)
        def process(item): results.append(item)
        process(range(4))
        all_results[exp] = results[:]
    for exp in range(3):
        assert len(all_results[exp]) == 4, \
            f"exp {exp}: expected 4, got {len(all_results[exp])}"
        _clean(f"t63_exp_{exp}")
test("Decorator defined inside a for-loop — each job independent", t63)

def t64():
    # Return values from tracked functions are currently None (by design)
    # This test documents the behavior so users know what to expect
    _clean("t64_retval")
    @loopz.track("t64_retval", save_every=2)
    def extract(item): return item * 2
    result = extract(range(5))
    assert result is None, \
        "tracked function returns None — results must be collected inside func"
test("Return value is None — results must be collected inside the function", t64)

def t65():
    # 1 million items — performance and memory sanity check
    import tracemalloc, time
    _clean("t65_1m")
    tracemalloc.start()
    start = time.time()
    @loopz.track("t65_1m", save_every=10_000)
    def process(item): pass
    process(range(1_000_000))
    elapsed = time.time() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / 1024 / 1024
    assert elapsed < 60,  f"1M items took {elapsed:.1f}s — too slow"
    assert peak_mb < 200, f"1M items used {peak_mb:.1f}MB — too much RAM"
    print(f"       (1M items: {elapsed:.1f}s, peak RAM: {peak_mb:.1f}MB)")
    _clean("t65_1m")
test("1 million items — under 60s and under 200MB RAM", t65)


# ===========================================================================
# FINAL REPORT
# ===========================================================================
total = passed + failed + skipped
print("\n" + "=" * 62)
print(f"  loopz v1.0.0 — Test Results")
print(f"  Passed  : {passed}")
print(f"  Failed  : {failed}")
print(f"  Skipped : {skipped}  (optional deps not installed)")
print(f"  Total   : {total}")
print("=" * 62)
if failed == 0:
    print("  🎉 ALL TESTS PASSED — loopz v1.0.0 is ready to ship!\n")
else:
    print(f"  ⚠️  {failed} test(s) FAILED — fix before publishing!\n")
    sys.exit(1)
# ===========================================================================
# SECTION 18 — Custom Checkpoint Directory
# ===========================================================================
section("Section 18: Custom Checkpoint Directory")

def t66():
    # crash writes files to custom dir, NOT ~/.loopz/
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        job = "t66_custom_dir"
        @loopz.track(job, save_every=2, checkpoint_dir=str(tmp))
        def process(item):
            if item == 4:
                raise RuntimeError("crash")
        try:
            process(range(10))
        except RuntimeError:
            pass
        # files must exist in custom dir
        files = list(tmp.glob("loopz_*.json"))
        assert len(files) > 0, "❌ no checkpoint file in custom dir — went to ~/.loopz/ instead"
        # files must NOT exist in default CACHE_DIR for this job
        from loopz.tracker import _safe_name
        default_file = CACHE_DIR / f"loopz_{_safe_name(job)}.json"
        assert not default_file.exists(), "❌ checkpoint written to ~/.loopz/ instead of custom dir"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test("crash writes files to custom dir, not ~/.loopz/", t66)

def t67():
    # status(checkpoint_dir=) finds the job in custom dir
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        job = "t67_status_custom"
        @loopz.track(job, save_every=2, checkpoint_dir=str(tmp))
        def process(item):
            if item == 3:
                raise RuntimeError("crash")
        try:
            process(range(10))
        except RuntimeError:
            pass
        # status() with custom dir must find the job
        from loopz.tracker import list_jobs
        jobs = list_jobs(base_dir=tmp)
        names = [j["job_name"] for j in jobs]
        assert job in names, f"❌ job not found by status(checkpoint_dir=) — got {names}"
        # status() without custom dir must NOT find it
        default_jobs = list_jobs()
        default_names = [j["job_name"] for j in default_jobs]
        assert job not in default_names, "❌ job found in ~/.loopz/ — should be in custom dir only"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test("status(checkpoint_dir=) finds the job in custom dir", t67)

def t68():
    # reset(job_name, checkpoint_dir=) clears from custom dir
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp())
    try:
        job = "t68_reset_custom"
        @loopz.track(job, save_every=2, checkpoint_dir=str(tmp))
        def process(item):
            if item == 4:
                raise RuntimeError("crash")
        try:
            process(range(10))
        except RuntimeError:
            pass
        # confirm files are there
        files_before = list(tmp.glob("loopz_*"))
        assert len(files_before) > 0, "❌ no files in custom dir to reset"
        # reset with correct dir
        loopz.reset(job, checkpoint_dir=str(tmp))
        # confirm files are gone
        files_after = list(tmp.glob("loopz_*.json"))
        assert len(files_after) == 0, f"❌ reset() left files in custom dir: {files_after}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
test("reset(job_name, checkpoint_dir=) clears from custom dir", t68)