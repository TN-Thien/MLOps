import os
import pickle
import random
import time
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import wandb
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CIFAR_ROOT = "Thuc_hanh_02/Bai_02/cifar-10-batches-py"
SELECTED_CLASSES = [1, 2, 6, 7, 8]
NUM_CLASSES = len(SELECTED_CLASSES)
TRAIN_PER_CLASS = 1000
USE_FULL_TEST_FOR_SELECTED_CLASSES = True  

BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_RUNS = 3

WANDB_PROJECT = "cifar10-augmentation"
WANDB_ENTITY = None

def load_cifar_batch(path: str):
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
    data = batch[b"data"]
    labels = batch[b"labels"]
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(labels, dtype=np.int64)
    return data, labels

def load_cifar10_raw(root: str):
    train_data_list = []
    train_labels_list = []
    for i in range(1, 6):
        batch_path = os.path.join(root, f"data_batch_{i}")
        data, labels = load_cifar_batch(batch_path)
        train_data_list.append(data)
        train_labels_list.append(labels)

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)

    test_path = os.path.join(root, "test_batch")
    test_data, test_labels = load_cifar_batch(test_path)

    meta_path = os.path.join(root, "batches.meta")
    label_names = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f, encoding="bytes")
        label_names = [x.decode("utf-8") for x in meta[b"label_names"]]

    return train_data, train_labels, test_data, test_labels, label_names

def select_subset(
    images: np.ndarray,
    labels: np.ndarray,
    selected_classes,
    train_per_class=1000,
    for_train=True
) -> Tuple[np.ndarray, np.ndarray]:
    label_to_new = {c: i for i, c in enumerate(selected_classes)}
    selected_indices = []

    if for_train:
        for c in selected_classes:
            idx = np.where(labels == c)[0]
            np.random.shuffle(idx)
            selected_indices.extend(idx[:train_per_class])
    else:
        if USE_FULL_TEST_FOR_SELECTED_CLASSES:
            for c in selected_classes:
                idx = np.where(labels == c)[0]
                selected_indices.extend(idx.tolist())
        else:
            for c in selected_classes:
                idx = np.where(labels == c)[0]
                np.random.shuffle(idx)
                selected_indices.extend(idx[:train_per_class])

    selected_indices = np.array(selected_indices)
    subset_images = images[selected_indices]
    subset_labels = labels[selected_indices]

    mapped_labels = np.array([label_to_new[int(l)] for l in subset_labels], dtype=np.int64)

    return subset_images, mapped_labels

class CIFARSubsetDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        """
        images: numpy array (N, 32, 32, 3) uint8
        labels: numpy array (N,)
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (32, 32, 3)
        label = int(self.labels[idx])
        img = Image.fromarray(img)  # PIL image
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_transforms(augment: bool):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if not augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomResizedCrop(
                size=32,
                scale=(0.8, 1.2),
                ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform

def create_dataloaders(
    train_images, train_labels,
    test_images, test_labels,
    augment: bool
):
    train_tf, test_tf = get_transforms(augment)
    train_ds = CIFARSubsetDataset(train_images, train_labels, transform=train_tf)
    test_ds = CIFARSubsetDataset(test_images, test_labels, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, test_loader

def visualize_augmentations(train_images, train_labels):
    """
    Mỗi lớp lấy 1 ảnh, hiển thị:
        - Hàng trên: ảnh gốc
        - Hàng dưới: ảnh sau augment
    """
    print("Hiển thị ví dụ ảnh trước/sau tăng cường dữ liệu...")
    label_to_new = {c: i for i, c in enumerate(SELECTED_CLASSES)}

    chosen_indices = []
    for original_class in SELECTED_CLASSES:
        new_label = label_to_new[original_class]
        idx_candidates = np.where(train_labels == new_label)[0]
        chosen_indices.append(idx_candidates[0])

    aug_transform, _ = get_transforms(augment=True)

    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(3 * NUM_CLASSES, 6))

    for i, idx in enumerate(chosen_indices):
        img = train_images[idx]

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Class {i} - gốc")
        axes[0, i].axis("off")

        pil_img = Image.fromarray(img)
        aug_img_tensor = aug_transform(pil_img)
        aug_img = aug_img_tensor.clone()
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        aug_img = aug_img * std + mean
        aug_img = torch.clamp(aug_img, 0.0, 1.0)
        aug_img_np = aug_img.permute(1, 2, 0).numpy()
        axes[1, i].imshow(aug_img_np)
        axes[1, i].set_title(f"Class {i} - augment")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = None
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)
        return out

class SmallResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def create_model(num_classes=NUM_CLASSES) -> nn.Module:
    model = SmallResNet(num_classes=num_classes)
    return model

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device
):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_experiment(
    train_images, train_labels,
    test_images, test_labels,
    use_augmentation: bool,
    run_idx: int,
    base_seed: int = 42
) -> Dict[str, Any]:
    config_name = "augmented" if use_augmentation else "original"

    seed = base_seed + run_idx
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = create_dataloaders(
        train_images, train_labels,
        test_images, test_labels,
        augment=use_augmentation
    )

    model = create_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    wandb_config = {
        "config_name": config_name,
        "use_augmentation": use_augmentation,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": seed,
        "num_classes": NUM_CLASSES,
    }

    run_name = f"{config_name}_run{run_idx+1}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config=wandb_config,
        reinit=True
    )

    best_val_acc = 0.0
    best_epoch = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    total_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        epoch_time = time.time() - epoch_start

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
        })

        print(
            f"[{config_name}][Run {run_idx+1}] "
            f"Epoch {epoch:02d}/{NUM_EPOCHS} - "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f} | "
            f"best_acc: {best_val_acc:.4f} (epoch {best_epoch})"
        )

    total_time = time.time() - total_start

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["total_train_time"] = total_time
    wandb.finish()

    metrics = {
        "best_val_acc": best_val_acc,
        "final_val_acc": history["val_acc"][-1],
        "best_epoch": best_epoch,
        "total_time": total_time,
        "history": history,
    }
    return metrics

def main():
    print("Đọc dữ liệu CIFAR-10 gốc...")
    train_data, train_labels, test_data, test_labels, label_names = load_cifar10_raw(CIFAR_ROOT)

    if label_names is not None:
        print("Label names:", label_names)
    print("Train raw shape:", train_data.shape)
    print("Test raw shape:", test_data.shape)

    print("Chọn subset 5 lớp và 5000 ảnh train...")
    subset_train_images, subset_train_labels = select_subset(
        train_data, train_labels, SELECTED_CLASSES, train_per_class=TRAIN_PER_CLASS, for_train=True
    )
    subset_test_images, subset_test_labels = select_subset(
        test_data, test_labels, SELECTED_CLASSES, train_per_class=TRAIN_PER_CLASS, for_train=False
    )

    print("Subset train shape:", subset_train_images.shape, subset_train_labels.shape)
    print("Subset test shape:", subset_test_images.shape, subset_test_labels.shape)

    visualize_augmentations(subset_train_images, subset_train_labels)

    all_results = {
        "original": [],
        "augmented": [],
    }

    print("Huấn luyện với dữ liệu GỐC (không augment)...")
    for run_idx in range(NUM_RUNS):
        metrics = run_experiment(
            subset_train_images, subset_train_labels,
            subset_test_images, subset_test_labels,
            use_augmentation=False,
            run_idx=run_idx,
            base_seed=42
        )
        all_results["original"].append(metrics)

    print("Huấn luyện với dữ liệu TĂNG CƯỜNG...")
    for run_idx in range(NUM_RUNS):
        metrics = run_experiment(
            subset_train_images, subset_train_labels,
            subset_test_images, subset_test_labels,
            use_augmentation=True,
            run_idx=run_idx,
            base_seed=1234
        )
        all_results["augmented"].append(metrics)

    def summarize(result_list):
        best_accs = [r["best_val_acc"] for r in result_list]
        final_accs = [r["final_val_acc"] for r in result_list]
        best_epochs = [r["best_epoch"] for r in result_list]
        times = [r["total_time"] for r in result_list]

        return {
            "mean_best_acc": float(np.mean(best_accs)),
            "std_best_acc": float(np.std(best_accs)),
            "mean_final_acc": float(np.mean(final_accs)),
            "std_final_acc": float(np.std(final_accs)),
            "mean_best_epoch": float(np.mean(best_epochs)),  # tốc độ hội tụ
            "std_best_epoch": float(np.std(best_epochs)),
            "mean_total_time": float(np.mean(times)),
            "std_total_time": float(np.std(times)),
        }

    summary_original = summarize(all_results["original"])
    summary_augmented = summarize(all_results["augmented"])

    print("\nTÓM TẮT KẾT QUẢ (TRUNG BÌNH 3 LẦN)\n")
    print(">>Cấu hình 1: DỮ LIỆU GỐC (KHÔNG AUGMENT)")
    for k, v in summary_original.items():
        print(f"{k}: {v:.4f}")

    print("\n>>Cấu hình 2: DỮ LIỆU TĂNG CƯỜNG")
    for k, v in summary_augmented.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
