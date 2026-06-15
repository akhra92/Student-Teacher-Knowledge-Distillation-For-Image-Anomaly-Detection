"""Streamlit app for student-teacher anomaly detection.

Pick a trained model, upload an image, and get an anomaly score plus a
heatmap of where the student diverges from the teacher. Each checkpoint
embeds the config it was trained with, so the app rebuilds the exact same
network and preprocessing without needing the YAML on disk.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

# ViT pos-embed resampling lacks some MPS kernels; fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from stad.eval import score_batch
from stad.models import build_networks
from stad.utils import resolve_device, set_seed

OUTPUTS_DIR = Path(__file__).parent / "outputs"


# --------------------------------------------------------------------------- #
# Preprocessing — mirrors stad/cli/score.py so inference matches training.
# --------------------------------------------------------------------------- #
def build_transform(cfg: dict) -> transforms.Compose:
    name = cfg["data"]["dataset_name"]
    imagenet = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if name == "cifar10":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            imagenet,
        ])
    if name == "mvtec":
        size = int(cfg["data"]["mvtec_img_size"])
    else:  # mnist / fashionmnist
        size = int(cfg["data"].get("vit_img_size", 224))
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        imagenet,
    ])


def output_size(cfg: dict, x: torch.Tensor) -> int:
    if cfg["data"]["dataset_name"] == "mvtec":
        return int(cfg["data"]["mvtec_img_size"])
    return x.shape[-1]


# --------------------------------------------------------------------------- #
# Model discovery / loading.
# --------------------------------------------------------------------------- #
def discover_checkpoints() -> dict[str, Path]:
    """Map a human label -> best.pth path for every trained experiment."""
    found: dict[str, Path] = {}
    if not OUTPUTS_DIR.exists():
        return found
    for ckpt in sorted(OUTPUTS_DIR.glob("*/checkpoints/best.pth")):
        found[ckpt.parent.parent.name] = ckpt
    return found


@st.cache_resource(show_spinner="Loading model…")
def load_model(ckpt_path: str, device_pref: str):
    """Build teacher+student from the config embedded in the checkpoint."""
    device = resolve_device(device_pref)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError(
            "Checkpoint is missing its embedded config; retrain to embed one."
        )
    cfg = ckpt["config"]
    state = ckpt["student"] if "student" in ckpt else ckpt
    threshold = ckpt.get("threshold")
    if threshold is not None and threshold != threshold:  # NaN
        threshold = None

    set_seed(int(cfg.get("seed", 42)))
    teacher, student, layer_indices = build_networks(cfg, device=device)
    student.load_state_dict(state)
    student.eval()
    return teacher, student, layer_indices, cfg, threshold, device


# --------------------------------------------------------------------------- #
# Heatmap — per-layer student/teacher feature distance, upsampled & summed.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_heatmap(student, teacher, x, layer_indices, out_size) -> np.ndarray:
    pred = student(x)
    real = teacher(x)
    heat: torch.Tensor | None = None
    for k in layer_indices:
        yp, y = pred[k], real[k]
        if yp.dim() == 4:
            d = ((yp - y) ** 2).mean(dim=1, keepdim=True)
        else:
            d = ((yp - y) ** 2).mean(dim=-1)
            n_tokens = d.shape[1]
            side = int(round((n_tokens - 1) ** 0.5))
            if side * side == n_tokens - 1:
                d = d[:, 1:].reshape(d.shape[0], 1, side, side)
            else:
                side = int(round(n_tokens ** 0.5))
                d = d.reshape(d.shape[0], 1, side, side)
        d = F.interpolate(d, size=(out_size, out_size), mode="bilinear", align_corners=False)
        heat = d if heat is None else heat + d
    heat = heat.squeeze().cpu().numpy()
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-12)
    return heat


def overlay_heatmap(base: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a 'jet' colormap of `heat` over the resized base image."""
    size = heat.shape[0]
    base_rgb = np.asarray(base.convert("RGB").resize((size, size))).astype(np.float32) / 255.0
    colored = matplotlib.colormaps["jet"](heat)[..., :3]  # (H, W, 3)
    blended = (1 - alpha) * base_rgb + alpha * colored
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
# UI.
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="STAD Anomaly Detection", page_icon="🔍", layout="wide")
st.title("🔍 Student-Teacher Anomaly Detection")
st.caption(
    "Knowledge-distillation anomaly detector. A student ViT is trained to mimic a "
    "frozen DINOv2 teacher on normal images only; at test time, regions where the "
    "student fails to match the teacher score as anomalous."
)

checkpoints = discover_checkpoints()

with st.sidebar:
    st.header("⚙️ Settings")
    if not checkpoints:
        st.error(f"No trained models found under `{OUTPUTS_DIR}`.")
        st.stop()

    model_name = st.selectbox("Trained model", list(checkpoints.keys()))

    device_options = ["auto", "cpu"]
    if torch.cuda.is_available():
        device_options.insert(1, "cuda")
    if torch.backends.mps.is_available():
        device_options.insert(1, "mps")
    device_pref = st.selectbox("Device", device_options)

    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.5, 0.05)

teacher, student, layer_indices, cfg, threshold, device = load_model(
    str(checkpoints[model_name]), device_pref
)

with st.sidebar:
    st.markdown("---")
    st.markdown("**Model info**")
    st.write(
        {
            "dataset": cfg["data"]["dataset_name"],
            "normal_class": cfg["data"]["normal_class"],
            "teacher": cfg["teacher"]["timm_model"],
            "layers": layer_indices,
        }
    )
    st.markdown("---")
    st.markdown("**Decision threshold**")
    if threshold is not None:
        st.success(f"Calibrated threshold: {threshold:.4f}")
        manual_threshold = float(threshold)
    else:
        st.info("This checkpoint has no calibrated threshold. Set one manually to get a verdict.")
        use_manual = st.checkbox("Use manual threshold", value=False)
        manual_threshold = (
            st.slider("Threshold", 0.0, 5.0, 1.0, 0.01) if use_manual else None
        )

uploaded = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded is None:
    st.info("👆 Upload an image to score it for anomalies.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
tfm = build_transform(cfg)
x = tfm(img).unsqueeze(0).to(device)

with st.spinner("Scoring…"):
    score = score_batch(
        student, teacher, x,
        layer_indices=layer_indices,
        lamda=float(cfg["train"]["lamda"]),
        direction_only=bool(cfg["train"].get("direction_loss_only", False)),
    ).item()
    heat = compute_heatmap(student, teacher, x, layer_indices, output_size(cfg, x))

overlay = overlay_heatmap(img, heat, alpha=alpha)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input")
    st.image(img, use_container_width=True)
with col2:
    st.subheader("Anomaly heatmap")
    st.image(overlay, use_container_width=True)

st.markdown("### Result")
m1, m2 = st.columns(2)
m1.metric("Anomaly score", f"{score:.4f}")
if manual_threshold is not None:
    is_anom = score >= manual_threshold
    m2.metric("Verdict", "ANOMALOUS" if is_anom else "normal", f"threshold {manual_threshold:.3f}")
    (st.error if is_anom else st.success)(
        f"{'⚠️ Anomaly detected' if is_anom else '✅ Looks normal'} "
        f"(score {score:.4f} {'≥' if is_anom else '<'} threshold {manual_threshold:.3f})"
    )
else:
    m2.metric("Verdict", "—")
    st.caption("Higher score = more anomalous. Set a threshold in the sidebar for a verdict.")
