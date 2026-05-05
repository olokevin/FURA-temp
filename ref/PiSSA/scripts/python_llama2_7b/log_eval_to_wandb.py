"""Parse evalplus output from a tee'd eval log and log pass@1 metrics to wandb.

evalplus prints lines like:

    HumanEval (base tests)
    pass@1: 0.213
    HumanEval+ (base + extra tests)
    pass@1: 0.171
    Mbpp (base tests)
    pass@1: 0.357
    Mbpp+ (base + extra tests)
    pass@1: 0.282

We pick those four numbers up via regex, write them to a metrics.json next to
the log, and log a single wandb run with those four scalar metrics.
"""
import argparse
import json
import os
import re
from pathlib import Path

import wandb


def parse_eval_log(log_path: Path) -> dict:
    text = log_path.read_text()
    # evalplus prints sections like:
    #   humaneval (base tests)
    #   pass@1:\t0.207
    #   humaneval+ (base + extra tests)
    #   pass@1:\t0.189
    # Match (case-insensitively) the section header on its own line, then the
    # *next* pass@1 line — this avoids accidentally pairing "[3/4] evalplus
    # pass@1 on HumanEval..." headers with the wrong score.
    pattern = re.compile(
        r"^(humaneval\+?|mbpp\+?)\s*\([^)]*\)\s*\n[^\n]*?pass@1:\s*([0-9.]+)",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    key_map = {
        "humaneval": "humaneval/pass@1",
        "humaneval+": "humaneval_plus/pass@1",
        "mbpp": "mbpp/pass@1",
        "mbpp+": "mbpp_plus/pass@1",
    }
    metrics = {}
    for match in pattern.finditer(text):
        section = match.group(1).lower()
        score = float(match.group(2))
        if section in key_map:
            metrics[key_map[section]] = score
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--eval_log", required=True)
    ap.add_argument("--metrics_out", required=True)
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()

    metrics = parse_eval_log(Path(args.eval_log))
    if not metrics:
        raise RuntimeError(
            f"Failed to parse any pass@1 score from {args.eval_log}; "
            "check the eval log for evalplus errors."
        )

    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    print("metrics:", json.dumps(metrics, indent=2))

    run = wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="eval",
        config={"ckpt": args.ckpt, "eval_log": args.eval_log},
        reinit=True,
    )
    wandb.log(metrics)
    run.summary.update(metrics)
    run.finish()
    print(f"Logged to wandb project={args.project} run={args.run_name}")


if __name__ == "__main__":
    main()
