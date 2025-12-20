from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from training.src.dataset import CauHinhDuLieu, TapDuLieuIQA
from training.src.model import MoHinhIQA

def dat_thiet_lap_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def danh_gia(model: torch.nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        p = model(x).squeeze(1)  # [B]
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(p.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    try:
        plcc = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        plcc = float("nan")
    try:
        srocc = float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        srocc = float("nan")

    return {"mae": mae, "rmse": rmse, "plcc": plcc, "srocc": srocc}

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
    p = argparse.ArgumentParser(description="Huấn luyện mô hình IQA (local MLOps)")
    p.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "mobilenet_v2"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--ten-model", default="iqa_viet_hoa", help="Tên model trong MLflow Registry")
    p.add_argument("--alias-thu-nghiem", default="thu_nghiem")
    p.add_argument("--alias-san-xuat", default="san_xuat")

    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    p.add_argument("--experiment", default="IQA_VietHoa")

    return p.parse_args()

def main() -> None:
    args = parse_args()
    dat_thiet_lap_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    cfg = CauHinhDuLieu()
    ds_train = TapDuLieuIQA(split=cfg.ten_split_train, cfg=cfg)
    ds_val = TapDuLieuIQA(split=cfg.ten_split_val, cfg=cfg)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
                "du_lieu": str(cfg.file_csv),
                "thu_muc_anh": str(cfg.thu_muc_anh),
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

            print(f"[Epoch {epoch}] loss={loss:.5f} | " +
                  " | ".join([f"{k}={v:.4f}" for k, v in metrics_val.items()]))

            if metrics_val["srocc"] == metrics_val["srocc"] and metrics_val["srocc"] > best_srocc:
                best_srocc = metrics_val["srocc"]
                best_epoch = epoch
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=args.ten_model,
                )

        mlflow.log_metric("best_val_srocc", best_srocc)
        mlflow.log_metric("best_epoch", best_epoch)

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{args.ten_model}'")
        if not versions:
            raise RuntimeError("Không tìm thấy model version nào trong Registry sau khi log_model.")

        versions_sorted = sorted(versions, key=lambda v: int(v.creation_timestamp))
        latest = versions_sorted[-1]
        vnum = int(latest.version)

        client.set_registered_model_alias(args.ten_model, args.alias_thu_nghiem, vnum)
        # client.set_registered_model_alias(args.ten_model, args.alias_san_xuat, vnum)

        print(f"Đã đăng ký model '{args.ten_model}' version={vnum} và gán alias '{args.alias_thu_nghiem}'")

if __name__ == "__main__":
    main()