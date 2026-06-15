# Student-Teacher Anomaly Detection — v2

Modernized PyTorch implementation of the IEEE Access paper *"Extensive
knowledge distillation model: an end-to-end effective anomaly detection model
for real-time industrial applications"* (Rakhmonov et al., 2023). This is a
reorganized rewrite of the original repository with the following changes.

## Project layout

```
Student-Teacher-Anomaly-Detection-V2/
├── configs/
│   ├── config.yaml              # MNIST default (DINOv2 ViT-Small teacher)
│   ├── config_vit_base.yaml     # MNIST / DINOv2 ViT-Base teacher
│   └── mvtec_capsule_dino.yaml  # MVTec / DINOv2 teacher
├── stad/                        # importable package
│   ├── data/dataloader.py       # train/val/test loaders, no leakage
│   ├── models/
│   │   ├── teacher.py           # TimmTeacher (ViT/DINOv2)
│   │   ├── vit_student.py       # ViT cloner (for the timm/DINOv2 teacher)
│   │   └── factory.py           # build_networks()
│   ├── losses/distillation.py   # MseDirectionLoss / DirectionOnly / MseLoss
│   ├── eval/
│   │   ├── detection.py         # image-level AUROC
│   │   ├── localization.py      # feature-distance heatmaps
│   │   └── metrics.py           # AUROC, Youden's J, PRO
│   ├── utils/                   # config, device, logging, seed
│   └── cli/                     # command-line entry points
│       ├── train.py             # training loop with TensorBoard (stad-train)
│       ├── test.py              # detection / localization eval (stad-test)
│       ├── score.py             # single-image inference (stad-score)
│       ├── cache_teacher.py     # pre-compute teacher features (stad-cache-teacher)
│       └── baseline_patchcore.py# optional PatchCore baseline (stad-patchcore)
├── scripts/                     # thin back-compat wrappers around stad.cli
├── tests/                       # pytest smoke tests
├── pyproject.toml               # packaging + console entry points
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
# Editable install: pulls the deps and registers the stad-* commands.
# (torch/torchvision come with it; in Docker they ship with the base image.)
pip install -e .

# Train on MNIST (one-class anomaly: normal_class=3)
stad-train --config configs/config.yaml

# Evaluate the best checkpoint
stad-test --config configs/config.yaml

# Score a single image
stad-score --config configs/config.yaml --image path/to/img.png \
           --heatmap heatmap.png

# MVTec with DINOv2 teacher
stad-train --config configs/mvtec_capsule_dino.yaml

# Optional: PatchCore baseline on the same MVTec category (needs `pip install anomalib`)
stad-patchcore --config configs/config.yaml

# Run tests
pip install -e '.[dev]' && pytest tests/
```

## Web app (Streamlit)

An interactive UI for scoring images against any trained checkpoint:

```bash
streamlit run streamlit_app.py
```

Pick a trained model from the sidebar (auto-discovered under `outputs/`), then
either score one of the **bundled sample images** for that model (a normal and a
defective example per model, under `samples/`) or **upload your own**. The app
shows the anomaly score and a student/teacher divergence heatmap. Each
checkpoint carries the config it was trained with, so the app rebuilds the
matching network and preprocessing automatically. If a checkpoint has no
calibrated decision threshold, set one in the sidebar to get a normal/anomalous
verdict.

Every command is also runnable without installing, from the project root:
`python -m stad.cli.train --config configs/config.yaml` (same for `test`,
`score`, `cache_teacher`, `baseline_patchcore`). The old `python
scripts/train.py` paths still work as thin wrappers.

## Configuration cheatsheet

| Section | Key | What it does |
|---|---|---|
| `device` | `auto` / `cuda` / `mps` / `cpu` | Hardware selection. |
| `seed` | int | Global RNG seed. |
| `data.dataset_name` | `mnist` / `fashionmnist` / `cifar10` / `mvtec` | Picks loader + transforms. |
| `data.normal_class` | int / str | The "normal" class (string for MVTec). |
| `data.val_fraction` | float | Held-out slice of training data for model selection. |
| `teacher.kind` | `timm` | A timm ViT teacher (e.g. DINOv2). |
| `teacher.timm_model` | str | timm model name (e.g. `vit_small_patch14_dinov2.lvd142m`). |
| `teacher.feature_layers` | list[int] | ViT block indices to compare, `0..depth-1` (e.g. `[2,5,8,11]`). |
| `student.mlp_ratio` | float | ViT student capacity bottleneck (teacher uses 4.0; lower = stronger anomaly signal). |
| `train.lamda` | float | Weight on the MSE term (vs. cosine direction term). |
| `train.cache_teacher` | bool | Cache teacher features in memory once, reuse every epoch (disables train shuffling so batches stay aligned). |
| `eval.localization` | bool | Pixel-level eval (MVTec only). |
| `eval.localization_method` | `feature_dist` | How to make heatmaps. |

## Anomaly score formula

For an image `x`, let `f^s_k(x)` and `f^t_k(x)` be the student's and teacher's
feature maps at layer index `k ∈ feature_layers`:

```
score(x) = Σ_k [1 − cos(f^s_k(x), f^t_k(x))] + λ · Σ_k MSE(f^s_k(x), f^t_k(x))
```

Higher = more anomalous. ROC-AUC is computed over the test set. An operating
threshold is calibrated on the validation set at training time (Youden's J
when the val set contains anomalies, the 99th percentile of normal scores
otherwise) and stored in `best.pth` — `stad-score` uses it to print a
normal/ANOMALOUS verdict next to the raw score.

## Baseline comparison (optional)

`stad-patchcore` runs [anomalib](https://github.com/open-edge-platform/anomalib)'s
PatchCore on the same MVTec category and data root as your config, so its
image/pixel AUROC is directly comparable with `stad-test` output.
anomalib is intentionally **not** in `requirements.txt` (it pulls in
lightning and friends) — install it only if you want the comparison:

```bash
pip install anomalib            # or: pip install -e '.[baselines]'
stad-patchcore --config configs/config.yaml          # category from config
stad-patchcore --config configs/config.yaml --category hazelnut
```

Results and logs land in `outputs/patchcore_<category>/`.

## What's *not* included

- **Other baselines** (PaDiM, EfficientAD): these would need to actually be
  run to be meaningful and are out of scope here.
- **Pre-downloaded MVTec data**: the loader expects the standard MVTec-AD
  category structure under `data/mvtec/<category>/{train,test,ground_truth}/`.

## Citation

```
@article{rakhmonov2023extensive,
  author  = {Rakhmonov, Akhrorjon Akhmadjon Ugli and Subramanian, Barathi
             and Olimov, Bekhzod and Kim, Jeonghong},
  title   = {Extensive knowledge distillation model: An end-to-end effective
             anomaly detection model for real-time industrial applications},
  journal = {IEEE Access},
  year    = {2023},
  doi     = {10.1109/ACCESS.2023.3293108}
}
```
