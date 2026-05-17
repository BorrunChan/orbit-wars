"""
End-to-end remote GPU training pipeline.

Connects to user's GPU server, uploads data + training script, installs deps,
runs training, downloads the resulting value_4p_model.py.

Usage:
    .venv/bin/python scripts/remote_train.py
"""
import os
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
DATA_GZ = ROOT / "data" / "v4p_games.jsonl.gz"
TRAINER = ROOT / "scripts" / "train_value_4p.py"

HOST = "117.50.221.35"
USER = "zhh"
PWD = os.environ.get("RPW", "zhh+=123")

REMOTE_WORK = "~/orbit_wars_train"


def main():
    print(f"Local data: {DATA_GZ} ({DATA_GZ.stat().st_size/1e6:.1f} MB)")
    print(f"Connecting to {USER}@{HOST}...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PWD, timeout=20)
    sftp = client.open_sftp()
    print("Connected.\n")

    def run(cmd, timeout=300, log=True):
        if log:
            print(f"$ {cmd}")
        _, out, err = client.exec_command(cmd, timeout=timeout)
        # Stream output
        for line in out:
            line = line.rstrip()
            if line:
                print(f"  {line}")
        e = err.read().decode().rstrip()
        if e:
            print(f"  [stderr] {e[:500]}")

    # Step 1: prepare workdir (mirror local structure: scripts/ + data/)
    run(f"mkdir -p {REMOTE_WORK}/scripts {REMOTE_WORK}/data")

    # Resolve absolute path
    _, out, _ = client.exec_command(f"echo {REMOTE_WORK}")
    remote_work_abs = out.read().decode().strip()
    print(f"Remote work dir: {remote_work_abs}")

    # Step 2: upload files via SFTP
    print(f"\nUploading...")
    t0 = time.time()
    sftp.put(str(DATA_GZ), f"{remote_work_abs}/data/v4p_games.jsonl.gz")
    print(f"  data: {time.time()-t0:.1f}s")
    sftp.put(str(TRAINER), f"{remote_work_abs}/scripts/train_value_4p.py")
    print(f"  trainer: {time.time()-t0:.1f}s total")

    # Step 3: gunzip + check xgboost (already installed on this machine)
    print("\nUnzip + check env...")
    run(f"cd {remote_work_abs}/data && [ -f v4p_games.jsonl ] || gunzip -k -f v4p_games.jsonl.gz; ls -lah")
    run("python3 -c 'import xgboost; print(\"xgboost\", xgboost.__version__)'", timeout=30)

    # Step 4: ensure sklearn/numpy
    run("python3 -c 'import numpy, sklearn; print(\"numpy\", numpy.__version__, \"sklearn\", sklearn.__version__)'",
        timeout=60)

    # Step 5: train
    print("\n=== Training on GPU ===")
    t0 = time.time()
    cmd = (f"cd {remote_work_abs} && "
           f"CUDA_VISIBLE_DEVICES=0 python3 scripts/train_value_4p.py --gpu --trees 800 --depth 7 2>&1")
    run(cmd, timeout=1800)
    print(f"\nTraining wall: {time.time()-t0:.1f}s")

    # Step 6: download result
    print("\nDownloading value_4p_model.py ...")
    remote_model = f"{remote_work_abs}/data/value_4p_model.py"
    # Check it exists
    _, out, _ = client.exec_command(f"ls -lah {remote_model}")
    info = out.read().decode().strip()
    print(f"  {info}")
    local_model = ROOT / "value_4p_model.py"
    sftp.get(remote_model, str(local_model))
    print(f"Saved to {local_model} ({local_model.stat().st_size:,} bytes)")

    sftp.close()
    client.close()
    print("\nAll done. Run bench next.")


if __name__ == "__main__":
    main()
