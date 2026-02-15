import os
import subprocess
import sys

TARGETS = ["art_painting", "sketch"]


def run_train(run_id: int, ablation_noaug: bool) -> None:
    exp_root = (
        f"experiments_pacs_ablation_noaug_5run/run_{run_id}"
        if ablation_noaug
        else f"experiments_pacs_original_5run/run_{run_id}"
    )
    for target in TARGETS:
        cmd = [
            sys.executable,
            "train.py",
            "--target",
            target,
            "--network",
            "resnet18",
            "--epochs",
            "60",
            "--batch_size",
            "64",
            "--lr",
            "1e-3",
            "--lr_d",
            "1e-3",
            "--lr_c",
            "1e-4",
            "--lr_cp",
            "1e-4",
            "--lbd_c",
            "0.5",
            "--lbd_cp",
            "0.01",
            "--lbd_d",
            "0.05",
            "--num_workers",
            "0",
            "--exp_folder",
            exp_root,
        ]
        if ablation_noaug:
            cmd += ["--flip", "0", "--jitter", "0", "--min_scale", "1.0", "--max_scale", "1.0"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        run_tsne(exp_root, target)


def run_tsne(exp_root: str, target: str) -> None:
    ckpt = f"{exp_root}/resnet18/PACS/{target}/best_model.pth"
    out = f"{exp_root}/resnet18/PACS/{target}/tsne_by_domain.png"
    cmd = [sys.executable, "tsne_plot.py"]
    env = dict(os.environ)
    env["TSNE_CHECKPOINT"] = ckpt
    env["TSNE_OUT"] = out
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    # for run_id in [0, 1, 2, 3, 4]:
    #     run_train(run_id, ablation_noaug=False)
    #     run_train(run_id, ablation_noaug=True)

    run_train(5, ablation_noaug=False)
    run_train(5, ablation_noaug=True)

if __name__ == "__main__":
    main()
