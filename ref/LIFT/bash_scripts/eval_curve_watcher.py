#!/usr/bin/env python
"""Watch a training run's snap/ dir for rolling intermediate checkpoints and
evaluate each on HumanEval, logging pass@1 to the SAME wandb run (resumed) so
the metric overlays the training-loss curve vs step.

Protocol (matches finetune_q{lora,fura}.py `_write_snapshot`):
  <output_dir>/snap/LATEST         : text file holding the latest completed step N
  <output_dir>/snap/step_<N>/      : the checkpoint for step N
                                     (qfura: dense config.json+pytorch_model.bin+tokenizer;
                                      qlora: adapter_config.json+adapter+tokenizer)

For each new N: copy step_<N> aside (so the trainer may overwrite snap/ freely),
run eval_code.sh on the copy, parse HumanEval base pass@1, log to wandb at
step=N, then delete the aside copy. Stops when <output_dir>/last/ appears
(training finished) and no newer snapshot is pending, or on --max-steps reached.

Usage:
  CUDA_VISIBLE_DEVICES=4 python eval_curve_watcher.py \
      --output-dir <run output dir> \
      --wandb-run-id <id> --wandb-project mixtral_mode_code \
      --base-model mistralai/Mixtral-8x7B-v0.1 \
      --final-steps 819 [--poll-seconds 60]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

PROJECT_DIR = "/home/yequan/Project/lora/lora-without-regret"
EVAL_SH = os.path.join(PROJECT_DIR, "ref/LIFT/bash_scripts/eval_code.sh")


def parse_pass_at_1(eval_log_path):
    """Return (base, plus) HumanEval pass@1 floats from an eval_code.sh log,
    or (None, None) if not found. evalplus prints:
        humaneval (base tests)
        pass@1:\t0.512
        humaneval+ (base + extra tests)
        pass@1:\t0.433
    """
    if not os.path.exists(eval_log_path):
        return None, None
    with open(eval_log_path, errors="ignore") as f:
        text = f.read()
    base = plus = None
    m = re.search(r"humaneval \(base tests\).*?pass@1:\s*([0-9.]+)", text, re.S | re.I)
    if m:
        base = float(m.group(1))
    m = re.search(r"humaneval\+ \(base.*?pass@1:\s*([0-9.]+)", text, re.S | re.I)
    if m:
        plus = float(m.group(1))
    return base, plus


def evaluate_snapshot(ckpt_dir, base_model):
    """Run eval_code.sh on ckpt_dir; return (base, plus) pass@1."""
    env = dict(os.environ)
    env["base_model"] = base_model
    # The eval GPU may be SHARED with other users' jobs. The 4-bit model is only
    # ~25 GiB, so cap gpu_memory_utilization low enough that vLLM tolerates an
    # already-partly-used card (default 0.9 demands ~84 GiB free and fails when
    # another process holds even ~15 GiB). 0.55 -> ~51 GiB, ample for 25 GiB
    # weights + KV at max_model_len 2048. Overridable via the watcher's env.
    env.setdefault("GEN_GPU_MEM_UTIL", "0.55")
    # CUDA_VISIBLE_DEVICES is inherited from the watcher's environment.
    proc = subprocess.run(
        ["bash", EVAL_SH, f"CKPT={ckpt_dir}"],
        env=env, cwd=PROJECT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    # eval_code.sh writes eval.log under <ckpt>/humaneval/; but since we point it
    # at a dense/adapter dir directly, the humaneval/ dir is created under it.
    # Prefer that log; fall back to captured stdout.
    log_path = os.path.join(ckpt_dir, "humaneval", "eval.log")
    base, plus = parse_pass_at_1(log_path)
    if base is None:
        # fall back to stdout capture
        tmp = ckpt_dir + ".evalout.txt"
        with open(tmp, "w") as f:
            f.write(proc.stdout.decode(errors="ignore"))
        base, plus = parse_pass_at_1(tmp)
    return base, plus, proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--wandb-run-id", required=True)
    ap.add_argument("--wandb-project", default="mixtral_mode_code")
    ap.add_argument("--base-model", default="mistralai/Mixtral-8x7B-v0.1")
    ap.add_argument("--final-steps", type=int, required=True,
                    help="Total training steps; watcher exits after training "
                         "finishes (last/ exists) and this step is evaluated.")
    ap.add_argument("--poll-seconds", type=int, default=60)
    args = ap.parse_args()

    import wandb
    run = wandb.init(project=args.wandb_project, id=args.wandb_run_id,
                     resume="allow")
    # The training process advances this run's global wandb step continuously
    # (train_loss at step=completed_steps). wandb DROPS any log whose step is
    # <= the current max step, so a second process logging eval at step=100
    # while training is already at step=450 gets silently discarded. Use a
    # dedicated x-axis metric (eval_step) instead of the global step so the
    # HumanEval curve plots at the correct training step regardless of ordering.
    wandb.define_metric("eval_step")
    wandb.define_metric("humaneval_pass@1", step_metric="eval_step")
    wandb.define_metric("humaneval_plus_pass@1", step_metric="eval_step")
    print(f"[watcher] resumed wandb run {args.wandb_run_id} "
          f"(project {args.wandb_project})", flush=True)

    snap_root = os.path.join(args.output_dir, "snap")
    last_dir = os.path.join(args.output_dir, "last")
    evaluated = set()

    while True:
        latest_path = os.path.join(snap_root, "LATEST")
        pending = None
        if os.path.exists(latest_path):
            try:
                with open(latest_path) as f:
                    pending = int(f.read().strip())
            except (ValueError, OSError):
                pending = None

        if pending is not None and pending not in evaluated:
            step_dir = os.path.join(snap_root, f"step_{pending}")
            if os.path.isdir(step_dir):
                # Copy aside so the trainer may overwrite snap/ during eval.
                aside = os.path.join(snap_root, f"_eval_step_{pending}")
                if os.path.isdir(aside):
                    shutil.rmtree(aside, ignore_errors=True)
                try:
                    shutil.copytree(step_dir, aside)
                except (OSError, shutil.Error) as e:
                    print(f"[watcher] copy failed for step {pending}: {e}; retrying",
                          flush=True)
                    time.sleep(args.poll_seconds)
                    continue
                print(f"[watcher] evaluating step {pending} ...", flush=True)
                base, plus, rc = evaluate_snapshot(aside, args.base_model)
                if base is not None:
                    run.log({"eval_step": pending,
                             "humaneval_pass@1": base,
                             "humaneval_plus_pass@1": plus if plus is not None else float("nan")})
                    print(f"[watcher] step {pending}: HumanEval={base} "
                          f"HumanEval+={plus} (logged to wandb)", flush=True)
                else:
                    print(f"[watcher] step {pending}: eval produced no pass@1 "
                          f"(rc={rc}); skipping", flush=True)
                evaluated.add(pending)
                shutil.rmtree(aside, ignore_errors=True)

        # Exit condition: training finished and the final step is evaluated.
        if os.path.isdir(last_dir) and args.final_steps in evaluated:
            print("[watcher] training finished and final step evaluated; exiting.",
                  flush=True)
            break
        # Also exit if training finished and no snapshot is pending/newer.
        if os.path.isdir(last_dir) and (pending is None or pending in evaluated):
            # Give the final last/ checkpoint one eval if not already done.
            if args.final_steps not in evaluated:
                print(f"[watcher] training done; evaluating final last/ as step "
                      f"{args.final_steps}", flush=True)
                base, plus, rc = evaluate_snapshot(last_dir, args.base_model)
                if base is not None:
                    run.log({"eval_step": args.final_steps,
                             "humaneval_pass@1": base,
                             "humaneval_plus_pass@1": plus if plus is not None else float("nan")})
                    print(f"[watcher] final step {args.final_steps}: "
                          f"HumanEval={base} HumanEval+={plus}", flush=True)
                evaluated.add(args.final_steps)
            print("[watcher] done.", flush=True)
            break

        time.sleep(args.poll_seconds)

    run.finish()


if __name__ == "__main__":
    main()
