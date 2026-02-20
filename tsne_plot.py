import os
from types import SimpleNamespace

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.dataset import MyDataset, _dataset_info, get_val_transformer
from models import model_factory

# Default run setup.
DEFAULT_CHECKPOINT = "./experiments_pacs_original_5run/run_0/resnet18/PACS/art_painting/best_model.pth"
DEFAULT_BATCH_SIZE = 128
DEFAULT_NUM_WORKERS = 0
DEFAULT_MAX_POINTS = 2000
DEFAULT_SEED = 42
DEFAULT_PERPLEXITY = 30.0


class DomainFlagDataset(Dataset):
    # domain_flag: 0 = source, 1 = target
    def __init__(self, names, labels, domain_flag, transform, data_dir):
        self.base = MyDataset(names, labels, img_transformer=transform, data_dir=data_dir)
        self.domain_flag = domain_flag

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return image, label, self.domain_flag


def _build_model(ckpt_args, state_dict, device):
    # Rebuild the same backbone used during training
    network = ckpt_args.get("network", "resnet18")
    num_classes = int(ckpt_args.get("num_classes", 7))
    source_domains = ckpt_args.get("source", ["cartoon", "photo", "sketch"])
    num_domains = len(source_domains) if source_domains else 3
    main_model, _, _, _ = model_factory.get_network(network)(
        num_classes=num_classes, num_domains=num_domains, pretrained=False
    )
    main_model.load_state_dict(state_dict, strict=True)
    main_model.to(device)
    main_model.eval()
    return main_model


def _collect_features(model, ckpt_args, device):
    # Read data paths and domains from checkpoint args
    transform = get_val_transformer(SimpleNamespace(image_size=int(ckpt_args.get("image_size", 224))))
    data_dir = ckpt_args.get("data_dir", "./dataset")
    datalist_dir = ckpt_args.get("datalist_dir", "./datalist")
    dataset_name = ckpt_args.get("dataset", "PACS")
    source_domains = ckpt_args.get("source", ["cartoon", "photo", "sketch"])
    target_domain = ckpt_args.get("target", "art_painting")

    datasets = []

    # Source domains
    for domain in source_domains:
        txt = os.path.join(datalist_dir, dataset_name, f"{domain}_test.txt")
        names, labels = _dataset_info(txt)
        datasets.append(DomainFlagDataset(names, labels, 0, transform, data_dir))

    # Target domain
    target_txt = os.path.join(datalist_dir, dataset_name, f"{target_domain}_test.txt")
    names, labels = _dataset_info(target_txt)
    datasets.append(DomainFlagDataset(names, labels, 1, transform, data_dir))

    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=False,
    )

    feats = []
    domain_flags = []
    with torch.no_grad():
        for images, _, flags in loader:
            images = images.to(device)
            _, feature = model(images)
            feats.append(feature.cpu().numpy())
            domain_flags.append(flags.numpy())

    feats = np.concatenate(feats, axis=0)
    domain_flags = np.concatenate(domain_flags, axis=0)
    return feats, domain_flags, target_domain


def _subsample(feats, domain_flags):
    if feats.shape[0] <= DEFAULT_MAX_POINTS:
        return feats, domain_flags
    rng = np.random.default_rng(DEFAULT_SEED)
    idx = rng.choice(feats.shape[0], size=DEFAULT_MAX_POINTS, replace=False)
    return feats[idx], domain_flags[idx]


def _save_plot(embedding, domain_flags, out_path, target_domain):
    plt.figure(figsize=(7, 6), dpi=140)
    colors = np.where(domain_flags == 0, "#1f77b4", "#17becf")
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=8, alpha=0.8)
    plt.title("t-SNE by Domain (Source vs Target)")
    plt.xticks([])
    plt.yticks([])

    # legend
    source_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=6, label="source")
    target_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#17becf", markersize=6, label=target_domain)
    plt.legend(handles=[source_patch, target_patch], loc="best")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot: {out_path}")


def main():
    checkpoint_path = os.environ.get("TSNE_CHECKPOINT", DEFAULT_CHECKPOINT)
    out_path = os.environ.get("TSNE_OUT", os.path.join(os.path.dirname(checkpoint_path), "tsne_by_domain.png"))

    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "main_model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing key: main_model_state_dict")
    ckpt_args = ckpt.get("args", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model(ckpt_args, ckpt["main_model_state_dict"], device)
    feats, domain_flags, target_domain = _collect_features(model, ckpt_args, device)
    feats, domain_flags = _subsample(feats, domain_flags)

    tsne = TSNE(
        n_components=2,
        perplexity=min(DEFAULT_PERPLEXITY, max(5.0, feats.shape[0] / 3.0)),
        random_state=DEFAULT_SEED,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(feats)
    _save_plot(embedding, domain_flags, out_path, target_domain)


if __name__ == "__main__":
    main()
