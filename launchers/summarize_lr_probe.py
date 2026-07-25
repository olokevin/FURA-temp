#!/usr/bin/env python3
"""Summarize the short-horizon LR probes produced by `muon_lr_probe.sh`.

Reads every `<probe_root>/<method>-<opt>-lr_<lr>-probe<N>-seed_<S>/training.log`,
extracts the `Step: n, LR: x, Loss: y` lines, and prints a markdown table of the
tail-window mean loss (the ranking signal) plus a few checkpoints along the way.

    python launchers/summarize_lr_probe.py [probe_root] [--tail 100]
"""

import argparse
import os
import re
import sys

DEFAULT_ROOT = (
    "/data/yequan/fura/lift/commonsense/meta-llama/Meta-Llama-3-8B/_muon_lr_probe"
)

STEP_RE = re.compile(r"^\s*Step:\s*(\d+),\s*LR:\s*([0-9.eE+-]+),\s*Loss:\s*([0-9.eE+-]+)")
# `lr` itself contains a hyphen (e.g. 1e-3), so it has to be matched lazily.
DIR_RE = re.compile(
    r"^(?P<method>[^-]+)-(?P<opt>[^-]+)-lr_(?P<lr>.+?)-probe(?P<steps>\d+)(?P<sched>-[a-z_]+)?-seed_(?P<seed>\d+)$"
)


def read_curve(path):
    curve = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                curve.append((int(m.group(1)), float(m.group(3))))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    ap.add_argument("--tail", type=int, default=100,
                    help="Width of the trailing step window averaged for the ranking (default 100).")
    args = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(args.root)):
        m = DIR_RE.match(name)
        log = os.path.join(args.root, name, "training.log")
        if not m or not os.path.isfile(log):
            continue
        curve = read_curve(log)
        if not curve:
            rows.append((m.groupdict(), None, None, 0))
            continue
        last_step = curve[-1][0]
        tail = [loss for step, loss in curve if step > last_step - args.tail]
        rows.append((m.groupdict(), sum(tail) / len(tail), curve, last_step))

    if not rows:
        print(f"no probe runs found under {args.root}", file=sys.stderr)
        return 1

    def sort_key(row):
        meta = row[0]
        return (meta["method"], meta["opt"], -float(meta["lr"]))

    rows.sort(key=sort_key)

    print(f"probe root: {args.root}")
    print(f"ranking metric: mean train loss over the last {args.tail} steps\n")
    print("| method | optimizer | LR | steps done | loss@100 | loss@250 | tail loss |")
    print("| ------ | --------- | -- | ---------: | -------: | -------: | --------: |")
    for meta, tail_loss, curve, last_step in rows:
        if curve is None:
            print(f"| {meta['method']} | {meta['opt']} | {meta['lr']} | 0 | — | — | (no log) |")
            continue
        at = {step: loss for step, loss in curve}
        f = lambda s: f"{at[s]:.4f}" if s in at else "—"
        print(
            f"| {meta['method']} | {meta['opt']} | {meta['lr']} | {last_step} "
            f"| {f(100)} | {f(250)} | {tail_loss:.4f} |"
        )

    print()
    for method in sorted({r[0]["method"] for r in rows}):
        cand = [r for r in rows if r[0]["method"] == method and r[1] is not None]
        if not cand:
            continue
        best = min(cand, key=lambda r: r[1])
        print(f"best {method}: {best[0]['opt']} lr={best[0]['lr']} tail_loss={best[1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
