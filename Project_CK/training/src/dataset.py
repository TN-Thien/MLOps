from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class CauHinhDuLieu:
    thu_muc_anh: Path = Path("data/512x384")
    file_csv: Path = Path("data/koniq10k_distributions_sets.csv")
    khoa_split: str = "set"  # tên cột split (thường là 'set')
    ten_split_train: str = "training"
    ten_split_val: str = "validation"
    ten_split_test: str = "test"


def _tim_cot(df: pd.DataFrame, ung_vien: list[str]) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for u in ung_vien:
        if u.lower() in cols_lower:
            return cols_lower[u.lower()]
    raise ValueError(
        "Không tìm thấy cột phù hợp. Các cột hiện có: "
        + ", ".join(df.columns.astype(str).tolist())
    )


class TapDuLieuIQA(Dataset):
    """
    Dataset IQA đọc từ:
    - Ảnh: data/512x384/<ten_anh>
    - CSV: data/koniq10k_distributions_sets.csv

    CSV thường có:
    - cột tên ảnh: image_name / image / filename...
    - cột MOS: MOS / mos / score...
    - cột split: set (training/validation/test)
    """

    def __init__(
        self,
        split: str,
        cfg: CauHinhDuLieu = CauHinhDuLieu(),
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.cfg = cfg
        self.split = split

        df = pd.read_csv(cfg.file_csv)

        cot_anh = _tim_cot(df, ["image_name", "image", "filename", "img_name", "name"])
        cot_mos = _tim_cot(df, ["MOS", "mos", "score", "quality", "mean_opinion_score"])
        cot_split = _tim_cot(df, [cfg.khoa_split, "split"])

        df = df[df[cot_split].astype(str).str.lower() == split.lower()].copy()
        if len(df) == 0:
            raise ValueError(
                f"Split '{split}' không có dòng nào trong CSV. "
                f"Hãy kiểm tra cột split '{cot_split}'."
            )

        self.ds_anh = df[cot_anh].astype(str).tolist()
        self.ds_mos = df[cot_mos].astype(float).tolist()

        # Chuẩn hoá MOS: nếu giá trị dạng 0–100 thì đưa về 0–1 để train ổn định
        # (sau đó serving sẽ nhân lại 100)
        self.ds_mos = [m / 100.0 if m > 1.5 else m for m in self.ds_mos]

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((512, 384)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.ds_anh)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ten_anh = self.ds_anh[idx]
        duong_dan = self.cfg.thu_muc_anh / ten_anh
        if not duong_dan.exists():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {duong_dan}")

        img = Image.open(duong_dan).convert("RGB")
        x = self.transform(img)

        y = torch.tensor(self.ds_mos[idx], dtype=torch.float32)  # 0–1
        return x, y
