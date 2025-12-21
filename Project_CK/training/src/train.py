from __future__ import annotations

import argparse
import os
import platform
from dataclasses import asdict
from typing import Dict

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

    p.add_argument("--backbone", default="efficientnet_b0",
                   choices=["efficientnet_b0", "resnet18", "mobilenet_v2"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)

    default_workers = 0 if platform.system().lower().startswith("win") else 2
    p.add_argument("--num-workers", type=int, default=default_workers)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model-name", default="iqa_efficientnet_b0", help="Tên model trong MLflow Registry")
    p.add_argument("--alias", default="staging", help="Alias trỏ tới version muốn serving (vd: staging)")

    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment", default="IQA")

    return p.parse_args()

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

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda")
    )

    model = MoHinhIQA(backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    with mlflow.start_run(run_name=f"train_{args.backbone}"):
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
                "csv": str(cfg.file_csv),
                "image_dir": str(cfg.thu_muc_anh),
                **{f"cfg_{k}": v for k, v in asdict(cfg).items()},
            }
        )

        best_srocc = -1.0
        best_epoch = -1

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

            if metrics_val["srocc"] == metrics_val["srocc"] and metrics_val["srocc"] > best_srocc:
                best_srocc = metrics_val["srocc"]
                best_epoch = epoch

                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    name="model",
                    registered_model_name=args.model_name,
                )

        mlflow.log_metric("best_val_srocc", best_srocc)
        mlflow.log_metric("best_epoch", best_epoch)

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{args.model_name}'")
        if not versions:
            raise RuntimeError("Không tìm thấy model version nào trong Registry sau khi log_model.")

        versions_sorted = sorted(versions, key=lambda v: int(v.creation_timestamp))
        latest = versions_sorted[-1]
        vnum = int(latest.version)

        client.set_registered_model_alias(args.model_name, args.alias, vnum)
        print(f"Registered model '{args.model_name}' version={vnum} | alias '{args.alias}'")

if __name__ == "__main__":
    main()
