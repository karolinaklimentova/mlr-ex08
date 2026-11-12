"""
Task 4 - turn task4_results.json into the report tables.

Prints:
  * the seed-0 comparison table (every variant, three benchmarks),
  * the multi-seed table with mean +- standard deviation,
  * the noise floor implied by the seeds, so a "gain" can be compared against it.

    .venv/bin/python task4_summarize.py
"""

import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "task4_results.json")

# The order the report presents the rows in.
ORDER = [
    "baseline",
    "+color",
    "+transfer-frozen",
    "+transfer-full",
    "+transfer-aug-cosine",
    "+regularized",
    "+historical-data",
    "+transfer-cosine+hist",
    "+transfer-cosine+hist+color",
    "+transfer-cosine+hist+color+adamw",
]

BENCHMARKS = [
    ("primary_val_accuracy", "Standard val"),
    ("camera_val_accuracy", "Camera-like val"),
    ("historical_val_accuracy", "Historical val (OOD)"),
]


def pct(x):
    return f"{100 * x:.1f}"


def main():
    with open(RESULTS) as handle:
        results = json.load(handle)

    by_variant = defaultdict(list)
    for run in results:
        by_variant[run["variant"]].append(run)

    print(f"{len(results)} runs over {len(by_variant)} variants\n")

    # ---------------------------------------------------------------- seed 0
    print("## Seed 0 - one run per variant\n")
    header = ["variant", "train imgs", "params"] + [n for _, n in BENCHMARKS] + ["s"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name in ORDER:
        for run in sorted(by_variant.get(name, []), key=lambda r: r["seed"]):
            if run["seed"] != 0:
                continue
            cells = [
                f"`{name}`",
                str(run["train_samples"]),
                f"{run['trainable_parameters'] / 1e3:.0f}k",
            ]
            cells += [pct(run[key]) for key, _ in BENCHMARKS]
            cells.append(f"{run['seconds']:.0f}")
            print("| " + " | ".join(cells) + " |")

    # ---------------------------------------------------------- multi seed
    print("\n## Multi-seed - mean +- standard deviation\n")
    header = ["variant", "seeds", "runs"] + [n for _, n in BENCHMARKS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    spread = {}
    for name in ORDER:
        runs = by_variant.get(name, [])
        if not runs:
            continue
        seeds = sorted(r["seed"] for r in runs)
        cells = [f"`{name}`", f"{len(seeds)} ({','.join(str(s) for s in seeds)})", str(len(runs))]
        accs = []
        for key, _ in BENCHMARKS:
            values = np.array([100 * r[key] for r in runs])
            accs.append(values)
            if len(values) == 1:
                cells.append(f"{values[0]:.1f}")
            else:
                cells.append(f"{values.mean():.1f} ± {values.std(ddof=0):.1f}")
        spread[name] = accs
        print("| " + " | ".join(cells) + " |")

    # ------------------------------------------------- balanced accuracy
    print("\n## Balanced accuracy (mean per-class recall), multi-seed\n")
    header = ["variant", "seeds"] + [f"{n} (bal.)" for _, n in BENCHMARKS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name in ORDER:
        runs = by_variant.get(name, [])
        if not runs:
            continue
        cells = [f"`{name}`", str(len(runs))]
        for key, _ in BENCHMARKS:
            values = np.array([100 * r[key.replace("_accuracy", "_balanced_accuracy")] for r in runs])
            cells.append(f"{values[0]:.1f}" if len(values) == 1
                         else f"{values.mean():.1f} ± {values.std(ddof=0):.1f}")
        print("| " + " | ".join(cells) + " |")

    # ------------------------------------------------ per-class, camera too
    print("\n## Per-class recall on all three benchmarks, seed 0\n")
    header = ["variant", "benchmark"] + list(results[0]["historical_per_class"])
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name in ORDER:
        run = next((r for r in by_variant.get(name, []) if r["seed"] == 0), None)
        if run is None:
            continue
        for field, label in [("primary_recall", "standard"),
                             ("camera_recall", "camera-like"),
                             ("historical_recall", "historical")]:
            cells = [f"`{name}`", label]
            for v in run["historical_per_class"].values():
                cells.append(pct(v[field]) if v[field] is not None else "-")
            print("| " + " | ".join(cells) + " |")
    print("\nSupport (standard / camera-like / historical): " + ", ".join(
        f"{c} {v['primary_support']}/{v['camera_support']}/{v['historical_support']}"
        for c, v in results[0]["historical_per_class"].items()))

    # ------------------------------------------------------- noise floor
    print("\n## Noise floor\n")
    print("Standard deviation of a variant across its own seeds, in percentage points.\n")
    header = ["variant"] + [n for _, n in BENCHMARKS]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    floors = defaultdict(list)
    for name, accs in spread.items():
        if len(accs[0]) < 2:
            continue
        cells = [f"`{name}`"]
        for column, (key, _) in zip(accs, BENCHMARKS):
            sd = column.std(ddof=0)
            floors[key].append(sd)
            cells.append(f"{sd:.1f}")
        print("| " + " | ".join(cells) + " |")
    print()
    for key, label in BENCHMARKS:
        if floors[key]:
            print(f"* {label}: median across-seed SD **{np.median(floors[key]):.1f} pp**, "
                  f"max {max(floors[key]):.1f} pp")

    # --------------------------------------------------- deltas vs baseline
    print("\n## Change against the baseline (seed-matched)\n")
    base = by_variant.get("baseline", [])
    common_seeds = set(r["seed"] for r in base)
    print("| variant | " + " | ".join(n for _, n in BENCHMARKS) + " |")
    print("|" + "|".join(["---"] * (len(BENCHMARKS) + 1)) + "|")
    for name in ORDER:
        runs = by_variant.get(name, [])
        paired = [s for s in sorted(common_seeds)
                  if any(r["seed"] == s for r in runs) and any(r["seed"] == s for r in base)]
        if not paired or name == "baseline":
            continue
        cells = [f"`{name}`"]
        for key, _ in BENCHMARKS:
            deltas = []
            for seed in paired:
                a = next(r[key] for r in runs if r["seed"] == seed)
                b = next(r[key] for r in base if r["seed"] == seed)
                deltas.append(100 * (a - b))
            values = np.array(deltas)
            sign = "+" if values.mean() >= 0 else ""
            cells.append(f"{sign}{values.mean():.1f}"
                         + (f" ± {values.std(ddof=0):.1f}" if len(deltas) > 1 else ""))
        print("| " + " | ".join(cells) + " |")

    # ------------------------------------------------------- per class, best
    print("\n## Per-class recall on the historical (out-of-distribution) set, seed 0\n")
    header = ["variant"] + [f"{c}" for c in results[0]["historical_per_class"]]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for name in ORDER:
        run = next((r for r in by_variant.get(name, []) if r["seed"] == 0), None)
        if run is None:
            continue
        cells = [f"`{name}`"] + [
            pct(v["historical_recall"]) if v["historical_recall"] is not None else "-"
            for v in run["historical_per_class"].values()
        ]
        print("| " + " | ".join(cells) + " |")

    print("\nSupport: " + ", ".join(
        f"{c}={v['historical_support']}"
        for c, v in results[0]["historical_per_class"].items()))


if __name__ == "__main__":
    main()
