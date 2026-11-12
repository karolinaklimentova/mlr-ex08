# MLR Exercise 8 — Leaf Classifier in the Browser

Live in-browser leaf classification with a MobileNetV3-Small ONNX model. The camera
frames never leave the device: preprocessing and inference both happen in JavaScript
with [ONNX Runtime Web](https://onnxruntime.ai/web/), loaded from a CDN.

**Live site:** <https://karolinaklimentova.github.io/mlr-ex08/>

Open the live URL on a phone and tap **Start rear camera**, or open it on a desktop
and upload a photo. A QR code for the same URL is printed at the bottom of the page
(generated locally in SVG — no third-party QR service).

## The model

`leaf_model.onnx` is the best variant of a controlled comparison of ten training
recipes: `mobilenet_v3_small` pretrained on ImageNet, fully fine-tuned for 40 epochs
with geometric **and colour** augmentation, a cosine learning-rate schedule, and the
previous years' photographs added to the training split. The interface did not change
from the earlier model — same `1×3×96×96` input, same mean/std, same class order — so
this page works with either checkpoint. Measured on three held-out validation sets:

| validation set | previous model | this model |
|---|---|---|
| standard `val` split (62 images) | 82.3 % | **90.3 %** |
| camera-like views (186) | 62.4 % | **83.3 %** |
| previous years' photographs (120, out of distribution) | 60.0 % | **90.0 %** |

Same architecture and file size (6.39 MB), so inference cost is unchanged.

## What it does

1. Reads `model_info.json` and follows it exactly — nothing is hard-coded:

   | spec | value |
   |---|---|
   | `classes` | `apple`, `cherry`, `chestnut`, `maple` |
   | `input_size` | `96` |
   | `mean` | `[0.485, 0.456, 0.406]` |
   | `std` | `[0.229, 0.224, 0.225]` |

2. Takes the **central square** of the current frame (shown as a dashed overlay),
   resizes it to `96 × 96`, scales RGB to `[0, 1]`, standardises each channel with
   `(x - mean) / std`, and builds a **`1 × 3 × 96 × 96` float32 NCHW** tensor — the
   `image` input the model declares.
3. Runs the model client-side. The exported graph ends in a tensor named `logits`,
   i.e. **raw un-normalised scores**, so the page applies **softmax** before display.
4. Shows a live probability bar for every one of the four classes, plus the top
   prediction, the raw probability to six decimals, and the per-frame inference time.

## Camera notes

* `<video playsinline muted autoplay>` — required for inline playback in iOS Safari.
* `getUserMedia({ video: { facingMode: { ideal: "environment" } } })` — requests the
  rear camera, falling back gracefully if only a front camera exists.
* GitHub Pages serves over HTTPS, which is what `getUserMedia` requires.
* A **Upload photo** control is always available as a fallback: it runs the identical
  crop → resize → normalise → infer path on a still image, so it works even where the
  camera is unavailable or denied.

## Files

| file | purpose |
|---|---|
| `index.html` | the whole application (markup, styles, preprocessing, inference, QR generator) |
| `leaf_model.onnx` | the trained classifier (`image` → `logits`, `1×3×96×96` float32) |
| `model_info.json` | class names and preprocessing specification |
| `check_image.jpg` | verification image, `96 × 96` |
| `check_expected.json` | reference probabilities for `check_image.jpg` |

## Accuracy check

`check_image.jpg` is a `96 × 96` image, so the central crop and resize are the identity
and the file is a clean numerical probe. Headless Chrome was driven over the DevTools
protocol to push `check_image.jpg` through the real `<input type="file">` control, and
the resulting probabilities were compared with `check_expected.json`:

| class | browser | expected | abs diff |
|---|---|---|---|
| apple | 0.439218 | 0.439215 | 3.0e-06 |
| cherry | 0.560630 | 0.560633 | 3.0e-06 |
| chestnut | 0.000130 | 0.000130 | 7.0e-10 |
| maple | 0.000022 | 0.000022 | 7.1e-11 |

Agreement is far better than three decimal places (worst absolute difference
`3.0e-06`, execution provider `wasm`). The camera path is checked the same way: with
Chrome driven by a synthetic video device, the exact `96 × 96` pixels the page fed to
the model were read back out of the browser and re-scored by desktop ONNX Runtime —
the two agree to `2.6e-06`, so the residual difference against `check_expected.json`
on a live frame is the test video's 4:2:0 chroma subsampling, not the pipeline.

## Deployment

Static site, no build step. Pushed to `main`; GitHub Pages serves the repository root.
The ONNX weights are fetched relative to the page URL, so the app works unchanged at the
`/mlr-ex08/` project-page base path.
