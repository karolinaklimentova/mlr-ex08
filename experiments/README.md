# Task 4 — experiment code, results and artefacts

The comparison behind the model that `main` deploys: 10 training recipes × 3 seeds,
scored on three held-out validation benchmarks. Written as plain scripts on top of the
Task 3 notebook pipeline — no framework, no helper classes, every variant is one changed
entry in `variant_configs()`.

| file | what it is |
|---|---|
| `task4_experiments.py` | the whole experiment: data loading, leakage cleaning, the three benchmarks, `LeafNetBN` / `LeafNetBNGAP` / MobileNetV3 variants, the training loop, the CLI |
| `task4_summarize.py` | turns `task4_results.json` into the report tables (mean ± SD, noise floor, seed-matched deltas, per-class recall) |
| `task4_export.py` | exports a variant's saved weights to `leaf_model.onnx` + `model_info.json` + `check_expected.json` |
| `task4_results.json` | all 30 runs, every metric, per-class recall, confusion matrices, loss curves, timings |
| `task4_cleaning_report.json` | exactly which historical images were dropped and why |
| `task4_historical_clean.txt` | the 679 kept historical images, as relative paths |
| `task4_camera_val.pt` | the frozen camera-like benchmark, 186 views of the 62 validation images, generated once under `torch.manual_seed(1234)` |
| `task4_weights_+transfer-cosine+hist+color_seed0.pt` | the exact weights of the shipped model |

## Running it

These scripts resolve paths **relative to their own directory**, so to re-run anything
they have to sit in the project root next to the data, not in this folder:

```
mlr_ex08/
  data/leaves/{train,val}      # 244 / 62 images
  leaves/{train,val}           # 700 / 120 images, last year's photographs
  task4_experiments.py         # <- copy the scripts here
```

Then, from the project root, using the project's virtualenv:

```bash
./.venv/bin/python task4_experiments.py                       # seed 0, all variants
./.venv/bin/python task4_experiments.py --seeds 0 1 2         # the stability check
./.venv/bin/python task4_experiments.py --only "+color" --save-weights
./.venv/bin/python task4_summarize.py                         # print the tables
./.venv/bin/python task4_export.py --variant "+transfer-cosine+hist+color" --seed 0 \
    --out-dir "task 2f/ex08_for_agent"
```

Runs take 6–87 s each on Apple Silicon `mps`; the full 30-run grid is ≈ 16 minutes.

## Two things to know before re-running

* **The image datasets are not in this repository** — only the derived artefacts
  (`task4_camera_val.pt`, the cleaning report, the kept-file manifest) are here.
* **`task4_camera_val.pt` must not be regenerated casually.** Every reported number was
  measured on those exact pixels; deleting it changes the benchmark and invalidates the
  comparison. Rebuild it only with `--rebuild-camera-val` if that is what you intend.

The findings — including the two benchmarks where the shipped model and the
`+…+adamw` variant are a statistical tie, and the leakage found in `leaves/train` — are
discussed in `REPORT.md`, which is kept local and not published.
