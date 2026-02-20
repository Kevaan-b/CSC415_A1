#!/usr/bin/env python3
import subprocess
import sys

TARGETS = ["art_painting", "sketch", "cartoon", "photo"]
TSNE_DOMAINS = ["art_painting", "sketch", "cartoon", "photo"]


def run_train(run_id: int, ablation_noaug: bool) -> None:
    # Separate output roots for baseline vs ablation.
    exp_root = (
        f"results/ablation/run_{run_id}"
        if ablation_noaug
        else f"results/reproduction/run_{run_id}"
    )
    for target in TARGETS:
        cmd = [
            sys.executable,
            "train.py",
            "--target",
            target,
            "--exp_folder",
            exp_root,
        ]
        if ablation_noaug:
            # Disable augmentation for ablation.
            cmd += ["--flip", "0", "--jitter", "0", "--min_scale", "1.0", "--max_scale", "1.0"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        run_tsne(exp_root, target)


def run_tsne(exp_root: str, target: str) -> None:
    ckpt = f"{exp_root}/resnet18/PACS/{target}/best_model.pth"
    out = f"{exp_root}/resnet18/PACS/{target}/tsne_by_domain.png"
    cmd = [
        sys.executable,
        "tsne_plot.py",
        "--checkpoint",
        ckpt,
        "--out",
        out,
        "--domains",
        *TSNE_DOMAINS,
        "--plot_mode",
        "domain",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    # both configs for 3 runs
    for run_id in [0, 1, 2]:
        run_train(run_id, ablation_noaug=False)
        run_train(run_id, ablation_noaug=True)

if __name__ == "__main__":
    main()
