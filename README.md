# Student-Teacher Anomaly Detection — v2

Modernized PyTorch implementation of the IEEE Access paper *"Extensive
knowledge distillation model: an end-to-end effective anomaly detection model
for real-time industrial applications"* (Rakhmonov et al., 2023). This is a
reorganized rewrite of the original repository with the following changes.

## Project layout

```
Student-Teacher-Anomaly-Detection-V2/
├── configs/
│   ├── config.yaml              # MNIST default
│   ├── mvtec_capsule.yaml       # MVTec / VGG16 teacher
│   └── mvtec_capsule_dino.yaml  # MVTec / DINOv2 teacher
├── stad/                        # importable package
│   ├── data/dataloader.py       # train/val/test loaders, no leakage
│   ├── models/
│   │   ├── teacher.py           # Vgg16Teacher + TimmTeacher
│   │   ├── student.py           # cloner CNN
│   │   └── factory.py           # build_networks()
│   ├── losses/distillation.py   # MseDirectionLoss / DirectionOnly / MseLoss
│   ├── eval/
│   │   ├── detection.py         # image-level AUROC
│   │   ├── localization.py      # 4 saliency methods + feature-distance map
│   │   └── metrics.py           # AUROC, Youden's J, PRO
│   └── utils/                   # config, device, logging, seed
├── scripts/
│   ├── train.py                 # CLI training loop with AMP + TensorBoard
│   ├── test.py                  # detection / localization eval
│   ├── score.py                 # single-image inference
│   └── cache_teacher.py         # pre-compute teacher features
├── tests/                       # pytest smoke tests
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
pip install -r requirements.txt

# Train on MNIST (one-class anomaly: normal_class=3)
python scripts/train.py --config configs/config.yaml

# Evaluate the best checkpoint
python scripts/test.py --config configs/config.yaml

# Score a single image
python scripts/score.py --config configs/config.yaml --image path/to/img.png \
                       --heatmap heatmap.png

# MVTec with DINOv2 teacher
python scripts/train.py --config configs/mvtec_capsule_dino.yaml

# Run tests
pytest tests/
```

## Configuration cheatsheet

| Section | Key | What it does |
|---|---|---|
| `device` | `auto` / `cuda` / `mps` / `cpu` | Hardware selection. |
| `seed` | int | Global RNG seed. |
| `data.dataset_name` | `mnist` / `fashionmnist` / `cifar10` / `mvtec` | Picks loader + transforms. |
| `data.normal_class` | int / str | The "normal" class (string for MVTec). |
| `data.val_fraction` | float | Held-out slice of training data for model selection. |
| `teacher.kind` | `vgg16` / `timm` | VGG16 (default) or any timm backbone. |
| `teacher.feature_layers` | list[int] | Indices of layers to compare. |
| `student.equal_size` | bool | True = full-size mirror; False = compressed cloner. |
| `train.lamda` | float | Weight on the MSE term (vs. cosine direction term). |
| `train.cache_teacher` | bool | Cache teacher features once per epoch in memory. |
| `eval.localization` | bool | Pixel-level eval (MVTec only). |
| `eval.localization_method` | `feature_dist` / `gradients` / `smooth_grad` / `gbp` | How to make heatmaps. |

## Anomaly score formula

For an image `x`, let `f^s_k(x)` and `f^t_k(x)` be the student's and teacher's
feature maps at layer index `k ∈ feature_layers`:

```
score(x) = Σ_k [1 − cos(f^s_k(x), f^t_k(x))] + λ · Σ_k MSE(f^s_k(x), f^t_k(x))
```

Higher = more anomalous. ROC-AUC is computed over the test set; an operating
threshold can be picked via Youden's J in `stad.eval.metrics.pick_threshold_youden`.

## What's *not* included

- **Baselines comparison** (PaDiM, PatchCore, EfficientAD): these would need to
  actually be run to be meaningful and are out of scope here.
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
