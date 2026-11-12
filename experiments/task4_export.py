"""
Task 4 - export the chosen model the same way the notebook does in cell 2e.

The weights come from the file task4_experiments.py saved with --save-weights, so
the shipped network is exactly the one whose numbers are in the report.

    .venv/bin/python task4_export.py --variant "+transfer-cosine+hist" --seed 0 --out-dir "task 2f/ex08_for_agent"
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import task4_experiments as t4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default=None,
                        help="state file; defaults to task4_weights_<variant>_seed<seed>.pt")
    parser.add_argument("--out-dir", default=t4.ROOT)
    parser.add_argument("--check-image", default=os.path.join(t4.ROOT, "check_image.jpg"))
    args = parser.parse_args()

    weights = args.weights or os.path.join(
        t4.ROOT, f"task4_weights_{args.variant}_seed{args.seed}.pt")
    print(f"weights: {weights}")
    blob = torch.load(weights, weights_only=False)
    cfg = blob["config"]

    # Rebuild the same network and put the trained weights back into it.
    model = t4.build_net(cfg, blob["seed"])
    model.load_state_dict(blob["state_dict"])
    model.eval().cpu()
    print(f"variant {args.variant} seed {args.seed} | net {cfg['net']} | "
          f"{sum(p.numel() for p in model.parameters())} parameters")

    size = t4.IMAGE_SIZE
    probe = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        print("probe output shape:", tuple(model(probe).shape))

    out_dir = os.path.abspath(args.out_dir)
    onnx_path = os.path.join(out_dir, "leaf_model.onnx")

    # The same call as notebook cell 49.
    torch.onnx.export(model, (probe,), onnx_path,
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes=None, external_data=False)
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_info = {"classes": t4.CLASSES, "input_size": size, "mean": t4.mean, "std": t4.std}
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, onnx_path)
    print(f"{onnx_path}: {os.path.getsize(onnx_path) / 1e6:.2f} MB")

    with open(os.path.join(out_dir, "model_info.json"), "w") as handle:
        json.dump(model_info, handle)
    print("model_info.json:", json.dumps(model_info))

    # The verification probe: the same file as before, scored by the new network, so
    # check_expected.json always agrees with the shipped leaf_model.onnx.
    with Image.open(args.check_image) as raw:
        raw = raw.convert("RGB")
    x = t4.val_transform(raw).unsqueeze(0)
    with torch.no_grad():
        probabilities = model(x).softmax(dim=1)[0]
    expected = dict(zip(t4.CLASSES, probabilities.tolist()))
    with open(os.path.join(out_dir, "check_expected.json"), "w") as handle:
        json.dump(expected, handle, indent=1)
    print("check_expected.json:", json.dumps(expected))

    # And the raw logits, worth having in the report.
    with torch.no_grad():
        print("logits:", [round(float(v), 6) for v in model(x)[0]])


if __name__ == "__main__":
    main()
