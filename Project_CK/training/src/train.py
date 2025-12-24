from __future__ import annotations

import argparse
import copy
import os
import platform
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.models.signature import infer_signature
from torch.utils.data import DataLoader

from training.src.dataset import CauHinhDuLieu, TapDuLieuIQA
from training.src.model import MoHinhIQA

def dat_thiet_lap_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pearson_np(a: np.ndarray, b: np.ndarray) -> float:
    """Tính Pearson correlation bằng NumPy (không cần SciPy)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")

def spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    """Tính Spearman correlation: rank bằng Pandas rồi Pearson bằng NumPy."""
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return pearson_np(ra, rb)

@torch.no_grad()
def danh_gia(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        p = model(x).squeeze(1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(p.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    plcc = pearson_np(y_true, y_pred)
    srocc = spearman_np(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "plcc": plcc,
        "srocc": srocc,
        "mae_100": mae * 100.0,
        "rmse_100": rmse * 100.0,
    }

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x).squeeze(1)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huấn luyện mô hình IQA (MLOps local)")

    p.add_argument(
        "--backbone",
        default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet18", "mobilenet_v2"],
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    default_workers = 0 if platform.system().lower().startswith("win") else 2
    p.add_argument("--num-workers", type=int, default=default_workers)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model-name", default="iqa_efficientnet_b0", help="Tên model trong MLflow Registry")
    p.add_argument("--alias", default="staging", help="Alias dùng cho serving (vd: staging)")

    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment", default="IQA")

    p.add_argument("--primary-metric", default="srocc", choices=["srocc", "plcc"])
    return p.parse_args()

def _is_better(new_metrics: Dict[str, float], cur_metrics: Dict[str, float], primary: str) -> bool:
    """So sánh model mới vs model hiện tại (đang staging)."""
    new_primary = float(new_metrics.get(primary, float("-inf")))
    cur_primary = float(cur_metrics.get(primary, float("-inf")))

    if new_primary > cur_primary:
        return True
    if new_primary < cur_primary:
        return False

    new_rmse = float(new_metrics.get("rmse_100", float("inf")))
    cur_rmse = float(cur_metrics.get("rmse_100", float("inf")))
    return new_rmse < cur_rmse

def _get_alias_metrics(client: mlflow.tracking.MlflowClient, model_name: str, alias: str) -> Dict[str, float]:
    """Lấy metrics từ tags của version đang được alias trỏ tới."""
    try:
        v = client.get_model_version_by_alias(model_name, alias)
    except Exception:
        return {}

    def _to_float(x: Optional[str]) -> float:
        try:
            return float(x) if x is not None else float("nan")
        except Exception:
            return float("nan")

    return {
        "srocc": _to_float(v.tags.get("val_srocc")),
        "plcc": _to_float(v.tags.get("val_plcc")),
        "rmse_100": _to_float(v.tags.get("val_rmse_100")),
        "mae_100": _to_float(v.tags.get("val_mae_100")),
    }

def main() -> None:
    args = parse_args()
    dat_thiet_lap_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    cfg = CauHinhDuLieu()

    ds_train = TapDuLieuIQA(split=cfg.ten_split_train, cfg=cfg)

    try:
        ds_val = TapDuLieuIQA(split=cfg.ten_split_val, cfg=cfg)
        ten_val = cfg.ten_split_val
    except Exception:
        ds_val = TapDuLieuIQA(split=cfg.ten_split_test, cfg=cfg)
        ten_val = cfg.ten_split_test

    ds_test = None
    try:
        ds_test = TapDuLieuIQA(split=cfg.ten_split_test, cfg=cfg)
    except Exception:
        ds_test = None

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = None
    if ds_test is not None:
        test_loader = DataLoader(
            ds_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )

    model = MoHinhIQA(backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name=f"train_{args.backbone}") as run:
        mlflow.log_params(
            {
                "backbone": args.backbone,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "device": device,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "val_split": ten_val,
                "primary_metric": args.primary_metric,
                "csv": str(cfg.file_csv),
                "image_dir": str(cfg.thu_muc_anh),
                **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
            }
        )

        best_primary = float("-inf")
        best_epoch = -1
        best_state = None
        best_val_metrics: Optional[Dict[str, float]] = None

        for epoch in range(1, args.epochs + 1):
            loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            metrics_val = danh_gia(model, val_loader, device)

            mlflow.log_metric("loss_train", loss, step=epoch)
            for k, v in metrics_val.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            print(
                f"[Epoch {epoch}] loss={loss:.5f} | "
                + " | ".join([f"{k}={v:.4f}" for k, v in metrics_val.items()])
            )

            primary_val = float(metrics_val.get(args.primary_metric, float("nan")))
            if primary_val == primary_val and primary_val > best_primary:
                best_primary = primary_val
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                best_val_metrics = metrics_val.copy()

        if best_state is None or best_val_metrics is None:
            raise RuntimeError("Không tìm thấy best model trong quá trình train (best_state=None).")

        model.load_state_dict(best_state)

        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_primary", best_primary)
        mlflow.log_metric("best_val_srocc", float(best_val_metrics["srocc"]))
        mlflow.log_metric("best_val_plcc", float(best_val_metrics["plcc"]))

        if test_loader is not None:
            metrics_test = danh_gia(model, test_loader, device)
            for k, v in metrics_test.items():
                mlflow.log_metric(f"test_{k}", v)
            print("Test metrics:", metrics_test)

        input_example = np.random.randn(1, 3, 384, 384).astype(np.float32)
        with torch.no_grad():
            sample_out = model(torch.from_numpy(input_example).to(device)).cpu().numpy()
        signature = infer_signature(input_example, sample_out)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=f"iqa_{args.backbone}",
            input_example=input_example,
            signature=signature,
        )

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{args.model_name}'")
        if not versions:
            raise RuntimeError("Không tìm thấy model version nào trong Registry sau khi log_model.")

        latest = max(versions, key=lambda v: int(v.version))
        new_version = int(latest.version)


        client.set_model_version_tag(args.model_name, new_version, "val_srocc", str(best_val_metrics["srocc"]))
        client.set_model_version_tag(args.model_name, new_version, "val_plcc", str(best_val_metrics["plcc"]))
        client.set_model_version_tag(args.model_name, new_version, "val_mae_100", str(best_val_metrics["mae_100"]))
        client.set_model_version_tag(args.model_name, new_version, "val_rmse_100", str(best_val_metrics["rmse_100"]))
        client.set_model_version_tag(args.model_name, new_version, "best_epoch", str(best_epoch))
        client.set_model_version_tag(args.model_name, new_version, "val_split", ten_val)

        cur_metrics = _get_alias_metrics(client, args.model_name, args.alias)
        new_metrics = {
            "srocc": float(best_val_metrics["srocc"]),
            "plcc": float(best_val_metrics["plcc"]),
            "rmse_100": float(best_val_metrics["rmse_100"]),
            "mae_100": float(best_val_metrics["mae_100"]),
        }

        if _is_better(new_metrics, cur_metrics, primary=args.primary_metric):
            client.set_registered_model_alias(f"iqa_{args.backbone}", args.alias, new_version)
            print(
                f"Updated alias '{args.alias}' -> version {new_version} "
                f"({args.primary_metric} {new_metrics[args.primary_metric]:.4f} "
                f"vs current {cur_metrics.get(args.primary_metric, float('nan')):.4f})"
            )
        else:
            print(
                f"ℹKeep alias '{args.alias}' (new model not better). "
                f"New {args.primary_metric}={new_metrics[args.primary_metric]:.4f}, "
                f"Current {args.primary_metric}={cur_metrics.get(args.primary_metric, float('nan')):.4f}"
            )

        print(f"Run: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

if __name__ == "__main__":
    main()
