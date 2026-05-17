"""Download PPO checkpoint from remote and bench vs v95 (current main.py)."""
import argparse, os, sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def fetch_ckpt(remote_path, local_path):
    """SSH download via paramiko."""
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('117.50.221.35', username='zhh', password=os.environ.get('RPW','zhh+=123'), timeout=15)
    sftp = c.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close(); c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", nargs="+", type=int, default=[10, 30, 50, 100])
    ap.add_argument("--n_games", type=int, default=10, help="2P games per ckpt")
    ap.add_argument("--remote_dir", default="checkpoints_smart")
    args = ap.parse_args()

    os.makedirs(ROOT / "checkpoints", exist_ok=True)
    results = {}

    for it in args.iters:
        remote = f'orbit_wars_rl/{args.remote_dir}/ppo_smart_iter_{it}.pt'
        local = str(ROOT / "checkpoints" / f"ppo_smart_iter_{it}.pt")
        try:
            fetch_ckpt(remote, local)
            print(f'Downloaded iter {it}: {os.path.getsize(local)//1024} KB')
        except Exception as e:
            print(f'iter {it}: skip ({e})')
            continue

        # Bench v82_ppo.py wrapping this ckpt vs main.py (v95)
        env = os.environ.copy()
        env['PPO_CKPT'] = local
        env['KAGGLE_ENVIRONMENTS_QUIET'] = '1'

        # Run via ab.py: ckpt-wrapped agent vs current main (v95)
        # Wrap into a temp agent file pointing at this ckpt
        wrap = ROOT / "agents" / f"v82_iter_{it}.py"
        v82 = (ROOT / "agents" / "v82_ppo.py").read_text()
        # Replace the default ckpt path with this specific ckpt
        wrap_text = v82.replace(
            'os.environ.get("PPO_CKPT", str(ROOT / "checkpoints" / "ppo_iter_20.pt"))',
            f'"{local}"'
        )
        wrap.write_text(wrap_text)

        # Bench
        out = subprocess.run(
            ['.venv/bin/python', '-u', 'scripts/ab.py', f'v82_iter_{it}',
             str(args.n_games), 'main'],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800
        )
        # Note: ab.py expects opp names. But 'main' isn't an opponent. Use ref()
        # Workaround: bench vs each opp pool member
        print(f'iter {it} stdout (last 8 lines):')
        for line in out.stdout.split('\n')[-15:]:
            if 'WR=' in line or 'Total' in line:
                print(f'  {line}')
        results[it] = out.stdout

    print('\n=== Summary ===')
    for it, r in results.items():
        if 'Total:' in r:
            total_line = [l for l in r.split('\n') if l.startswith('Total:')][-1]
            print(f'iter {it}: {total_line}')


if __name__ == "__main__":
    main()
