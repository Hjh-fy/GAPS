import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="/home/gaps/GAPS/flower_runtime")
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--profile", default="strong_cls")
    parser.add_argument("--local-epochs", default="5")
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--num-classes", default="4")
    parser.add_argument("--input-dim", default="8")
    parser.add_argument("--num-clients", default="4")
    parser.add_argument("--num-phases", default="3")
    parser.add_argument(
        "--eval-split",
        choices=("calibration", "test"),
        default="test",
    )
    parser.add_argument(
        "--python-bin",
        default="/home/gaps/GAPS/gaps_rpi_env/bin/python",
    )
    parser.add_argument("--server-address", default="127.0.0.1:18080")
    args = parser.parse_args()

    project = Path(args.project)
    log_dir = project / args.log_root / args.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"client_{args.client_id}.log", "ab", buffering=0)

    cmd = [
        args.python_bin,
        "-m",
        "gaps_flower.client_app",
        "--server-address",
        args.server_address,
        "--client-id",
        args.client_id,
        "--run-tag",
        args.run_id,
        "--data-root",
        str(project / "dataset" / args.data_root),
        "--device",
        "cpu",
        "--local-epochs",
        args.local_epochs,
        "--batch-size",
        args.batch_size,
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
        "--eval-split",
        args.eval_split,
    ]

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
