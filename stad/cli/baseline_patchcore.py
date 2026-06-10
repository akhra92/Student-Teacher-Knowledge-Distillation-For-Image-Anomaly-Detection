"""Optional PatchCore baseline (via anomalib) for comparison with the student.

PatchCore is a strong memory-bank baseline for MVTec anomaly detection. This
script trains/evaluates it on the same MVTec category and data root as the
given config, so its image/pixel AUROC can be compared directly with
`stad-test` results.

anomalib is an *optional* dependency (it pulls in lightning and friends):

    pip install anomalib

Usage:
    stad-patchcore --config configs/config.yaml
    stad-patchcore --config configs/config.yaml --category hazelnut
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from stad.utils import load_config, set_seed, setup_logger


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--category", type=str, default=None,
                   help="MVTec category. Defaults to data.normal_class from the config.")
    p.add_argument("--data_root", type=str, default=None,
                   help="Defaults to data.data_root from the config.")
    return p.parse_args()


def _import_anomalib():
    """Import the anomalib pieces we need, tolerating 1.x/2.x API renames."""
    try:
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as e:
        sys.exit(
            "This baseline needs the optional dependency `anomalib` "
            f"(import failed: {e}).\nInstall it with:  pip install anomalib"
        )
    try:  # anomalib >= 2.0
        from anomalib.data import MVTecAD as MVTecData
    except ImportError:  # anomalib 1.x
        from anomalib.data import MVTec as MVTecData
    return Engine, Patchcore, MVTecData


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    category = args.category or data_cfg["normal_class"]
    if data_cfg["dataset_name"] != "mvtec" and args.category is None:
        sys.exit(
            "PatchCore baseline is MVTec-only, but the config uses "
            f"dataset_name={data_cfg['dataset_name']!r}. Pass --category explicitly "
            "to run it against an MVTec category anyway."
        )
    if not isinstance(category, str):
        sys.exit(f"MVTec category must be a string, got {category!r}.")

    mvtec_root = Path(args.data_root or data_cfg["data_root"]) / "mvtec"
    if not (mvtec_root / category).is_dir():
        sys.exit(f"MVTec category dir not found: {mvtec_root / category}")

    set_seed(int(cfg.get("seed", 42)))
    out_dir = Path(cfg["output_dir"]) / f"patchcore_{category}"
    log = setup_logger("baseline_patchcore", log_dir=out_dir / "logs")

    Engine, Patchcore, MVTecData = _import_anomalib()

    # Build the datamodule on the same data the student-teacher model sees.
    # Constructor kwargs differ across anomalib versions, so only pass the
    # ones this version accepts.
    dm_kwargs = {
        "root": str(mvtec_root),
        "category": category,
        "train_batch_size": int(cfg["train"]["batch_size"]),
        "eval_batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 4)),
        "image_size": (int(data_cfg["mvtec_img_size"]),) * 2,
        "seed": int(cfg.get("seed", 42)),
    }
    accepted = inspect.signature(MVTecData.__init__).parameters
    dropped = sorted(set(dm_kwargs) - set(accepted))
    dm_kwargs = {k: v for k, v in dm_kwargs.items() if k in accepted}
    if dropped:
        log.info(f"anomalib {MVTecData.__name__} does not accept {dropped}; using its defaults.")

    datamodule = MVTecData(**dm_kwargs)
    model = Patchcore()
    engine = Engine(default_root_dir=str(out_dir))

    log.info(f"Fitting PatchCore on MVTec/{category} (root={mvtec_root}) ...")
    engine.fit(model=model, datamodule=datamodule)

    log.info("Evaluating on the test set ...")
    results = engine.test(model=model, datamodule=datamodule)

    log.info("=" * 60)
    log.info(f"PatchCore baseline — MVTec/{category}")
    for res in results:
        for key, value in sorted(res.items()):
            log.info(f"  {key} = {value:.4f}" if isinstance(value, float) else f"  {key} = {value}")
    log.info(f"Compare with the student-teacher model via: stad-test --config {args.config}")


if __name__ == "__main__":
    main()
