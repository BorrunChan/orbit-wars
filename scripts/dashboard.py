"""
Local dashboard for monitoring orbit-wars training/benchmarking.

Serves http://localhost:8765 with auto-refreshing Chinese-language HTML.

Sections:
  - 系统状态: CPU, 内存, 运行中进程
  - 数据收集进度: 实时进度条
  - 数据资产: 所有 jsonl 文件
  - Agent 版本对比: 每个版本 vs 5 强对手的胜率表
  - 最近 bench 详情: 单独看每个 bench log
  - 模型文件: shot_model.py / value_4p_model.py 状态

Usage:
  .venv/bin/python scripts/dashboard.py [port=8765]
"""

import html
import http.server
import os
import re
import socketserver
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


def get_processes():
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        if "grep" in ln or "dashboard.py" in ln:
            continue
        if any(k in ln for k in ("tournament", "ab.py", "collect_4p", "collect_shots",
                                  "collect_diverse", "round_robin", "train_value",
                                  "remote_train", "search.py", "extract_shots")):
            parts = ln.split(None, 10)
            if len(parts) >= 11:
                rows.append({
                    "pid": parts[1], "cpu": parts[2], "mem": parts[3],
                    "start": parts[8], "time": parts[9],
                    "cmd": parts[10][:300],
                })
    return rows


def get_system_load():
    try:
        out = subprocess.run(["uptime"], capture_output=True, text=True).stdout
        m = re.search(r"load averages?:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
        if m:
            return {"load_1m": m.group(1), "load_5m": m.group(2), "load_15m": m.group(3)}
    except Exception:
        pass
    return None


def get_memory():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        free = re.search(r"Pages free:\s+(\d+)", out)
        active = re.search(r"Pages active:\s+(\d+)", out)
        if free and active:
            page = 16384
            return {
                "free_gb": int(free.group(1)) * page / 1e9,
                "active_gb": int(active.group(1)) * page / 1e9,
            }
    except Exception:
        pass
    return None


def parse_bench_log(path):
    """Parse ab.py / tournament_4p.py log."""
    if not os.path.exists(path):
        return None
    text = Path(path).read_text(errors="ignore")
    # ab.py rows: "  vs lb1224        [32m 7W  3L  WR=70%[0m"
    rows = []
    for ln in text.splitlines():
        m = re.search(r"vs (\S+)\s+\[3\dm\s*(\d+)W\s+(\d+)L\s+WR=(\d+)%", ln)
        if m:
            rows.append({"opp": m.group(1), "w": int(m.group(2)),
                         "l": int(m.group(3)), "wr": int(m.group(4))})
    total = None
    m1 = re.search(r"Total:\s+(\d+)/(\d+)\s+=\s+(\d+)%\s+time=(\d+)s", text)
    if m1:
        total = {"w": int(m1.group(1)), "g": int(m1.group(2)),
                 "pct": int(m1.group(3)), "time": int(m1.group(4))}
    else:
        m4 = re.search(r"Total:\s+(\d+)W\s+/\s+(\d+)L\s+=\s+(\d+)%\s+time=(\d+)s", text)
        if m4:
            w, l = int(m4.group(1)), int(m4.group(2))
            total = {"w": w, "g": w + l, "pct": int(m4.group(3)),
                     "time": int(m4.group(4))}
    # 4P tournament per-game rows
    games_4p = []
    for ln in text.splitlines():
        m = re.search(r"seed=(\d+) pos=(\d+)\s+\[3\dm(WIN|LOSS)\s*\[0m\s+rewards=(\[.*?\])\s+steps=(\d+)", ln)
        if m:
            games_4p.append({
                "seed": int(m.group(1)), "pos": int(m.group(2)),
                "result": m.group(3), "rewards": m.group(4),
                "steps": int(m.group(5)),
            })
    return {"rows": rows, "total": total, "games_4p": games_4p,
            "mtime": os.path.getmtime(path),
            "size": os.path.getsize(path)}


def get_recent_logs():
    out = []
    for pat in ("v*_bench.log", "v*_*.log", "*_bench*.log",
                "*4p*.log", "*weak*.log", "*confirm*.log"):
        out.extend(Path("/tmp").glob(pat))
    # dedupe + sort by recency
    seen = set()
    uniq = []
    for p in out:
        if p in seen or "dash" in p.name or "tail" in p.name:
            continue
        seen.add(p)
        uniq.append(p)
    uniq.sort(key=lambda p: -os.path.getmtime(p))
    return uniq[:18]


def get_data_files():
    out = []
    for name in ("v4p_games.jsonl", "diverse_games.jsonl",
                 "round_robin_games.jsonl", "shot_games.jsonl",
                 "shot_outcomes_v2.jsonl", "winner_traces.jsonl",
                 "value_4p_model.py", "shot_model.py"):
        for parent in (ROOT / "data", ROOT):
            p = parent / name
            if p.exists():
                if name.endswith(".jsonl"):
                    with open(p, "rb") as f:
                        lines = sum(1 for _ in f)
                else:
                    lines = None
                out.append({"name": name, "path": str(p.relative_to(ROOT)),
                             "lines": lines, "size": p.stat().st_size,
                             "mtime": p.stat().st_mtime})
                break
    return out


def get_collection_progress():
    out = []
    for log in Path("/tmp").glob("collect_*.log"):
        text = log.read_text(errors="ignore")
        last_seed = re.findall(r"\[seed (\d+)/(\d+).*?\] saved=(\d+).*?eta=(\d+)s", text)
        if last_seed:
            cur, total, saved, eta = last_seed[-1]
            done = "Done:" in text
            out.append({"file": log.name, "cur": int(cur), "total": int(total),
                         "saved": int(saved), "eta": int(eta), "done": done,
                         "mtime": log.stat().st_mtime})
    out.sort(key=lambda x: -x["mtime"])
    return out


def get_agent_versions():
    """List all agents/v*.py with timestamps."""
    out = []
    agents_dir = ROOT / "agents"
    if not agents_dir.exists():
        return out
    for p in agents_dir.glob("v*.py"):
        out.append({"name": p.stem, "size": p.stat().st_size,
                     "mtime": p.stat().st_mtime})
    out.sort(key=lambda x: x["name"])
    return out


def get_current_main():
    main_py = ROOT / "main.py"
    sub = ROOT / "submission.tar.gz"
    info = {}
    if main_py.exists():
        first_lines = main_py.read_text(errors="ignore").split("\n", 5)[:5]
        # Extract first docstring line
        for ln in first_lines:
            if ln.strip().startswith("Orbit Wars - Agent"):
                info["main_desc"] = ln.strip()
                break
        info["main_mtime"] = main_py.stat().st_mtime
        info["main_size"] = main_py.stat().st_size
    if sub.exists():
        info["submission_size"] = sub.stat().st_size
        info["submission_mtime"] = sub.stat().st_mtime
    return info


# ============= Remote GPU status (cached) =============
_REMOTE_CACHE = {"data": None, "ts": 0.0, "ttl": 8.0}


def get_remote_status():
    """SSH to remote GPU server, fetch RL training status. Cached 8s."""
    now = time.time()
    if _REMOTE_CACHE["data"] and now - _REMOTE_CACHE["ts"] < _REMOTE_CACHE["ttl"]:
        return _REMOTE_CACHE["data"]
    try:
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect("117.50.221.35", username="zhh",
                        password=os.environ.get("RPW", "zhh+=123"),
                        timeout=4)
        # Bundle everything into one command
        cmd = (
            "echo '===GPU==='; "
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader; "
            "echo '===PROCS==='; "
            "ps -ef | grep -E 'rl_|train_value|orbit_wars' | grep -v grep | awk '{print $2, $5, $7, $NF}' | head -10; "
            "echo '===LOG==='; "
            "tail -30 ~/orbit_wars_rl/rl_v0.log 2>/dev/null; "
            "echo '===CKPT==='; "
            "ls -lah ~/orbit_wars_rl/*.pt ~/orbit_wars_rl/checkpoints/ 2>/dev/null | head -10; "
        )
        _, out, _ = client.exec_command(cmd, timeout=8)
        text = out.read().decode()
        client.close()
        # Parse sections
        sections = {}
        cur = None
        for line in text.splitlines():
            if line.startswith("===") and line.endswith("==="):
                cur = line.strip("=").strip()
                sections[cur] = []
            elif cur:
                sections[cur].append(line)
        data = {
            "gpus": sections.get("GPU", []),
            "procs": sections.get("PROCS", []),
            "log_tail": sections.get("LOG", []),
            "ckpts": sections.get("CKPT", []),
            "err": None,
        }
    except Exception as e:
        data = {"gpus": [], "procs": [], "log_tail": [], "ckpts": [],
                "err": str(e)[:200]}
    _REMOTE_CACHE["data"] = data
    _REMOTE_CACHE["ts"] = now
    return data


def render_remote_section():
    """HTML section for remote GPU."""
    r = get_remote_status()
    if r["err"]:
        return (f'<div class=card style="background:#fff3e0">'
                f'⚠️ 远端连接失败: <code>{html.escape(r["err"])}</code></div>')
    # GPU table
    gpu_html = '<table style="font-size:12px"><tr><th>GPU</th><th>名称</th>'\
               '<th>利用率</th><th>显存</th><th>温度</th></tr>'
    for line in r["gpus"]:
        # format: index, name, util%, mem_used MiB, mem_total MiB, temp C
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 6:
            continue
        idx, name, util, mu, mt, temp = cells
        util_pct = util.replace("%", "").strip()
        try:
            util_int = int(util_pct.split()[0])
        except Exception:
            util_int = 0
        color = "#2e7d32" if util_int > 50 else ("#f57c00" if util_int > 10 else "#888")
        gpu_html += (f'<tr><td>{idx}</td><td><code>{html.escape(name)}</code></td>'
                     f'<td style="color:{color};font-weight:bold">{util}</td>'
                     f'<td>{mu} / {mt}</td><td>{temp}</td></tr>')
    gpu_html += '</table>'

    # Procs
    proc_html = '<table style="font-size:12px"><tr><th>PID</th><th>启动</th>'\
                '<th>CMD</th></tr>'
    if not r["procs"] or all(not p.strip() for p in r["procs"]):
        proc_html += '<tr><td colspan=3 style="color:#888">(无 RL 进程)</td></tr>'
    else:
        for line in r["procs"]:
            if not line.strip():
                continue
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue
            pid, start, _t, cmd = parts
            proc_html += (f'<tr><td><code>{html.escape(pid)}</code></td>'
                          f'<td>{html.escape(start)}</td>'
                          f'<td><code style="font-size:11px">{html.escape(cmd[:80])}</code></td></tr>')
    proc_html += '</table>'

    # Log tail
    log_text = "\n".join(r["log_tail"]) or "(无日志)"
    log_html = (f'<pre style="background:#1e1e1e;color:#d4d4d4;padding:8px;'
                f'border-radius:4px;font-size:11px;max-height:300px;overflow:auto">'
                f'{html.escape(log_text)}</pre>')

    # Checkpoints
    ckpt_text = "\n".join(r["ckpts"]) or "(无 checkpoint)"
    ckpt_html = (f'<pre style="background:#f5f5f5;padding:6px;'
                 f'font-size:11px">{html.escape(ckpt_text)}</pre>')

    return (f'<h3 style="margin-top:8px">🖥 GPU 状态</h3>{gpu_html}'
            f'<h3 style="margin-top:12px">⚙️ RL 进程</h3>{proc_html}'
            f'<h3 style="margin-top:12px">📜 训练日志 (尾 30 行)</h3>{log_html}'
            f'<h3 style="margin-top:12px">💾 Checkpoints</h3>{ckpt_html}')


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_age(mtime):
    age = time.time() - mtime
    if age < 60:
        return f"{int(age)} 秒前"
    if age < 3600:
        return f"{int(age/60)} 分钟前"
    if age < 86400:
        return f"{int(age/3600)} 小时前"
    return f"{int(age/86400)} 天前"


def fmt_time(t):
    return time.strftime("%m-%d %H:%M:%S", time.localtime(t))


def color_for_wr(wr):
    if wr >= 50:
        return "#2e7d32"
    if wr >= 30:
        return "#f57c00"
    return "#c62828"


def render_html():
    procs = get_processes()
    logs = get_recent_logs()
    data = get_data_files()
    coll = get_collection_progress()
    agents = get_agent_versions()
    main_info = get_current_main()
    remote_html = render_remote_section()
    load = get_system_load()
    mem = get_memory()

    # Process table
    proc_rows = ""
    if procs:
        for p in procs:
            cmd_short = html.escape(p["cmd"])
            proc_rows += (f"<tr><td><code>{p['pid']}</code></td>"
                           f"<td>{p['cpu']}%</td><td>{p['mem']}%</td>"
                           f"<td>{p['time']}</td>"
                           f"<td><code>{cmd_short}</code></td></tr>")
    else:
        proc_rows = '<tr><td colspan=5 style="color:#888">无</td></tr>'

    # System info
    sys_info = ""
    if load:
        sys_info += (f"系统负载 (1m/5m/15m): "
                      f"<strong>{load['load_1m']} / {load['load_5m']} / {load['load_15m']}</strong> &nbsp; ")
    if mem:
        sys_info += (f"内存: 可用 <strong>{mem['free_gb']:.1f} GB</strong>, "
                      f"使用 <strong>{mem['active_gb']:.1f} GB</strong>")

    # Collection progress
    coll_html = ""
    for c in coll:
        pct = int(c["cur"] / max(1, c["total"]) * 100)
        bar_color = "#4caf50" if c["done"] else "#2196f3"
        status = "✓ 完成" if c["done"] else f"运行中, ETA {c['eta']}秒"
        coll_html += (f'<div style="margin:8px 0">'
                       f'<strong>{c["file"]}</strong> '
                       f'— seed {c["cur"]}/{c["total"]} ({pct}%), '
                       f'已存 {c["saved"]} 局 — <span style="color:#666">{status}</span>'
                       f'<div style="background:#eee;height:8px;width:400px;border-radius:4px;margin-top:4px;overflow:hidden">'
                       f'<div style="background:{bar_color};height:8px;width:{pct*4}px"></div>'
                       f'</div></div>')
    if not coll_html:
        coll_html = '<p style="color:#888">无活跃收集任务</p>'

    # Data file table
    data_rows = ""
    for d in sorted(data, key=lambda x: -x["mtime"]):
        lines_str = f"{d['lines']:,}" if d["lines"] is not None else "—"
        data_rows += (f"<tr><td><code>{d['path']}</code></td>"
                       f"<td>{lines_str}</td><td>{fmt_size(d['size'])}</td>"
                       f"<td>{fmt_age(d['mtime'])}</td></tr>")
    if not data_rows:
        data_rows = '<tr><td colspan=4 style="color:#888">无</td></tr>'

    # Bench results: parse all logs, group by name pattern
    bench_results = []
    for log in logs:
        parsed = parse_bench_log(str(log))
        if parsed and (parsed["total"] or parsed["rows"]):
            bench_results.append({"name": log.name, **parsed})

    bench_html = ""
    for b in bench_results[:15]:
        total = b.get("total")
        rows = b.get("rows", [])
        games_4p = b.get("games_4p", [])

        if total:
            total_color = color_for_wr(total["pct"])
            header = (f'<div style="display:flex;justify-content:space-between;'
                       f'align-items:baseline;flex-wrap:wrap">'
                       f'<strong style="font-size:1.05em">{b["name"]}</strong>'
                       f'<div><span style="color:{total_color};font-size:1.25em;font-weight:bold">'
                       f'{total["w"]}/{total["g"]} = {total["pct"]}%</span>'
                       f' <small style="color:#888">耗时 {total["time"]}s · {fmt_age(b["mtime"])}</small></div>'
                       f'</div>')
        else:
            header = f'<strong>{b["name"]}</strong> <small>{fmt_age(b["mtime"])}</small>'

        bench_html += f'<div style="border:1px solid #ddd;padding:10px;margin:6px 0;border-radius:6px;background:#fafafa">{header}'

        if rows:
            bench_html += '<table style="margin-top:6px;width:100%"><tr>'
            for r in rows:
                rc = color_for_wr(r["wr"])
                bench_html += (f'<td style="padding:4px 8px;border-right:1px dashed #ccc;text-align:center">'
                                f'<small style="color:#555">{r["opp"]}</small><br>'
                                f'<span style="color:{rc};font-weight:bold">{r["w"]}W {r["l"]}L</span><br>'
                                f'<span style="color:{rc}">{r["wr"]}%</span></td>')
            bench_html += '</tr></table>'

        if games_4p:
            wins = sum(1 for g in games_4p if g["result"] == "WIN")
            bench_html += (f'<div style="margin-top:8px"><small>'
                            f'4P 每局: <strong>{wins}胜 / {len(games_4p)-wins}负</strong></small><br>')
            # Last 10 game results as colored squares
            bench_html += '<div style="display:flex;flex-wrap:wrap;gap:2px;margin-top:4px">'
            for g in games_4p:
                bg = "#4caf50" if g["result"] == "WIN" else "#e57373"
                bench_html += (f'<span title="seed={g["seed"]} pos={g["pos"]} steps={g["steps"]}" '
                                f'style="background:{bg};color:white;padding:1px 6px;'
                                f'border-radius:3px;font-size:11px;font-family:monospace">'
                                f's{g["seed"]}p{g["pos"]}</span>')
            bench_html += '</div></div>'

        bench_html += '</div>'

    if not bench_html:
        bench_html = '<p style="color:#888">尚无 bench 结果</p>'

    # Agent versions
    agent_html = ""
    cur_main_size = main_info.get("main_size")
    for a in agents[-15:]:
        is_current = cur_main_size and abs(a["size"] - cur_main_size) < 50
        marker = " ←当前 main.py" if is_current else ""
        agent_html += (f'<tr><td><code>{a["name"]}</code></td>'
                        f'<td>{fmt_size(a["size"])}</td>'
                        f'<td>{fmt_age(a["mtime"])}</td>'
                        f'<td>{marker}</td></tr>')

    main_summary = ""
    if main_info:
        if "main_desc" in main_info:
            main_summary += f'当前 <code>main.py</code>: <strong>{html.escape(main_info["main_desc"])}</strong><br>'
        if "submission_size" in main_info:
            main_summary += (f'<code>submission.tar.gz</code>: '
                              f'{fmt_size(main_info["submission_size"])} '
                              f'({fmt_age(main_info["submission_mtime"])})')

    return f"""<!doctype html>
<html lang=zh-CN><head>
<meta charset=utf-8>
<title>Orbit Wars 训练监控</title>
<meta http-equiv=refresh content=5>
<style>
body {{ font-family: -apple-system, "PingFang SC", sans-serif; max-width: 1400px;
       margin: 16px auto; padding: 0 20px; color: #222 }}
h1 {{ font-size: 1.4em; color: #1565c0; margin: 8px 0 }}
h2 {{ font-size: 1.05em; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px;
      margin-top: 20px; color: #424242 }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px }}
td, th {{ padding: 5px 8px; text-align: left; border-bottom: 1px solid #eee }}
th {{ background: #f5f5f5; color: #555 }}
code {{ font-family: ui-monospace, "Menlo", monospace; font-size: 12px; color: #444 }}
small {{ color: #888 }}
.card {{ background: white; border: 1px solid #e0e0e0; padding: 10px;
         margin: 8px 0; border-radius: 6px }}
.banner {{ padding: 8px 12px; background: #e3f2fd; border-left: 4px solid #1976d2;
           margin: 12px 0; font-size: 13px }}
</style></head><body>

<h1>🚀 Orbit Wars 训练监控</h1>
<small>每 5 秒自动刷新 — 当前时间 {time.strftime('%Y-%m-%d %H:%M:%S')}</small>

<div class=banner>
{main_summary or '当前 main.py 未识别'}
</div>

<h2>💻 系统状态</h2>
<p>{sys_info or '无数据'}</p>

<h2>🔄 运行中进程 ({len(procs)} 个)</h2>
<table>
<tr><th>PID</th><th>CPU</th><th>MEM</th><th>已运行</th><th>命令</th></tr>
{proc_rows}
</table>

<h2>📦 数据收集进度</h2>
{coll_html}

<h2>💾 数据 & 模型文件</h2>
<table>
<tr><th>文件</th><th>记录数</th><th>大小</th><th>修改时间</th></tr>
{data_rows}
</table>

<h2>🏆 最近 Bench 结果 (按时间倒序，{len(bench_results)} 条)</h2>
<small style="color:#888">绿色 = WR ≥ 50%, 黄 = 30-50%, 红 = &lt; 30%</small>
{bench_html}

<h2>☁️ 云端 GPU RL 训练</h2>
<small style="color:#888">SSH 117.50.221.35 · 缓存 8s · 看 ~/orbit_wars_rl/rl_v0.log</small>
{remote_html}

<h2>🧪 Agent 版本历史 ({len(agents)} 个)</h2>
<table>
<tr><th>版本</th><th>大小</th><th>修改时间</th><th></th></tr>
{agent_html}
</table>

<p style="margin-top:40px;color:#aaa;font-size:11px;text-align:center">
Orbit Wars Dashboard · 5s auto-refresh · stop with Ctrl-C
</p>

</body></html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            page = render_html()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode())
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.send_response(500)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"Error: {e}\n\n{tb}".encode())

    def log_message(self, *a, **kw):
        pass


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main():
    print(f"Dashboard at http://localhost:{PORT}/")
    with ReusableTCPServer(("127.0.0.1", PORT), Handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    main()
