import argparse
import os
import subprocess
from pathlib import Path


def client_paths(data_root: str, client_ids: str) -> str:
    root = Path(data_root)
    if not root.is_absolute():
        root = Path("dataset") / root
    return ",".join(
        str(root / f"client_{cid}")
        for cid in client_ids.split(",")
        if cid
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="/root/GAPS")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-clients", required=True)
    parser.add_argument("--target-clients", required=True)
    parser.add_argument("--rounds", default="25")
    parser.add_argument("--profile", default="strong_cls")
    parser.add_argument("--num-classes", default="4")
    parser.add_argument("--input-dim", default="8")
    parser.add_argument("--num-clients", default="4")
    parser.add_argument("--num-phases", default="3")
    parser.add_argument("--server-address", default="0.0.0.0:8080")
    parser.add_argument("--python-bin", default="/root/gaps_env/bin/python")
    parser.add_argument(
        "--da-mode",
        choices=("legacy_strong", "corrected_b2"),
        default="legacy_strong",
    )
    parser.add_argument("--target-ce-weight", type=float, default=0.0)
    args = parser.parse_args()

    project = Path(args.project)
    output_dir = project / args.results_root / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    min_clients = str(len([c for c in args.source_clients.split(",") if c]))
    cmd = [
        args.python_bin,
        "-m",
        "gaps_flower.server_app",
        "--server-address",
        args.server_address,
        "--rounds",
        args.rounds,
        "--min-clients",
        min_clients,
        "--strategy",
        "gaps",
        "--profile",
        args.profile,
        "--num-classes",
        args.num_classes,
        "--input-dim",
        args.input_dim,
        "--num-clients",
        args.num_clients,
        "--num-phases",
        args.num_phases,
        "--run-name",
        args.run_id,
        "--output-dir",
        f"{args.results_root}/{args.run_id}",
        "--save-history",
        "true",
        "--use-domain-adapt",
        "true",
        "--server-val-data",
        client_paths(args.data_root, args.source_clients),
        "--server-calib-data",
        client_paths(args.data_root, args.target_clients),
        "--domain-adapt-steps",
        "100",
        "--domain-adapt-warmup",
        "0",
        "--strict-calibration-split",
        "true",
        "--da-device",
        "cpu",
        "--use-adapted-as-global",
        "true",
        "--da-lambda-target-ce",
        str(args.target_ce_weight),
        "--da-lambda-proto",
        "0.05",
        "--da-lambda-consistency",
        "2.0",
        "--da-lambda-residual",
        "0.1",
        "--da-target-ce-label-smoothing",
        "0.0",
        "--da-target-ce-class-balanced",
        "false",
        "--da-server-opt-lr",
        "0.0005",
    ]
    if args.da_mode == "legacy_strong":
        cmd.extend(
            [
                "--da-preset",
                "fixed_da_strong",
                "--use-proto-mmd",
                "true",
                "--da-use-coral",
                "true",
                "--da-use-mmd",
                "true",
                "--da-use-adversarial",
                "true",
                "--da-coral-class-conditional",
                "true",
                "--da-lambda-coral",
                "0.5",
                "--da-lambda-global-mmd",
                "0.5",
                "--da-lambda-class-mmd",
                "0.5",
                "--da-lambda-proto-anchor",
                "0.3",
                "--da-lambda-adv",
                "0.5",
                "--da-lambda-proto-mmd",
                "0.2",
                "--da-lambda-stage-mmd",
                "0.2",
                "--da-mmd-objective",
                "legacy_quartic",
                "--da-stage-alignment",
                "legacy_intra_domain",
                "--da-adv-feature-objective",
                "legacy_grl_plus",
            ]
        )
    else:
        cmd.extend(
            [
                "--da-preset",
                "none",
                "--use-proto-mmd",
                "false",
                "--da-use-coral",
                "false",
                "--da-use-mmd",
                "true",
                "--da-use-adversarial",
                "false",
                "--da-coral-class-conditional",
                "true",
                "--da-lambda-coral",
                "0.0",
                "--da-lambda-global-mmd",
                "0.5",
                "--da-lambda-class-mmd",
                "0.5",
                "--da-lambda-proto-anchor",
                "0.3",
                "--da-lambda-adv",
                "0.0",
                "--da-lambda-proto-mmd",
                "0.0",
                "--da-lambda-stage-mmd",
                "0.0",
                "--da-mmd-objective",
                "mmd2",
                "--da-stage-alignment",
                "cross_domain_same_class_phase",
                "--da-adv-feature-objective",
                "wasserstein_min",
            ]
        )

    log = open(output_dir / "server_launch.log", "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        close_fds=True,
        env=os.environ.copy(),
    )
    print(proc.pid)


if __name__ == "__main__":
    main()
