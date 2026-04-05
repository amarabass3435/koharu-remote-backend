"""
Build the Koharu Colab backend notebook (koharu_backend.ipynb).

Run:  python colab/build_notebook.py
Produces: colab/koharu_backend.ipynb
"""

import json, os

# ── helper ────────────────────────────────────────────────────────────
def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}

def code(source: str):
    return {"cell_type": "code", "metadata": {}, "source": _lines(source),
            "outputs": [], "execution_count": None}

def _lines(s: str):
    """Split into the line-list format .ipynb expects (each line ends with \\n except the last)."""
    lines = s.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result

# ── cells ─────────────────────────────────────────────────────────────

TITLE = md("""\
# 🌸 Koharu — Remote GPU Backend (Colab)

This notebook runs the **Koharu** manga-translation server on a Colab GPU
and exposes it to your local PC via a free **Cloudflare Tunnel**.

### How to use
1. **Runtime → Change runtime type → GPU** (T4 is fine, A100 is faster).
2. Run every cell **top-to-bottom**.
3. Copy the `*.trycloudflare.com` URL printed by Cell 5.
4. Open your local Koharu GUI and point it at that URL (Settings → Backend URL).
5. Use Koharu normally — detect, OCR, inpaint all run on this Colab GPU.

> Translation (LLM) is handled by whichever provider you configure in the GUI
> (MiniMax, OpenAI, local, etc.). This notebook only handles the **vision pipeline**.\
""")

CELL0 = code("""\
#@title 🔍 Cell 0 — Check GPU & install system deps
import subprocess, os

# Verify GPU
gpu_info = subprocess.check_output(["nvidia-smi"], text=True)
print(gpu_info)

# System packages needed for the koharu binary
subprocess.run(
    ["apt-get", "update", "-qq"],
    check=True, stdout=subprocess.DEVNULL
)
subprocess.run(
    ["apt-get", "install", "-y", "-qq",
     "libwebkit2gtk-4.1-dev", "libappindicator3-dev",
     "librsvg2-dev", "patchelf", "curl", "wget"],
    check=True, stdout=subprocess.DEVNULL
)
print("✅ System deps installed.")\
""")

CELL1 = code("""\
#@title 📥 Cell 1 — Download latest Koharu Linux binary from GitHub Releases
import urllib.request, json, os, subprocess, stat

REPO = "mayocream/koharu"
API  = f"https://api.github.com/repos/{REPO}/releases/latest"

print("Fetching latest release info …")
req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json"})
with urllib.request.urlopen(req) as r:
    release = json.loads(r.read())

tag = release["tag_name"]
print(f"Latest release: {tag}")

# Find the plain Linux binary (not .deb, not .AppImage)
asset_url = None
for asset in release["assets"]:
    name = asset["name"]
    # The tauri-action uploadPlainBinary produces a file like "koharu" or "koharu-<tag>-linux"
    if "linux" in name.lower() and not name.endswith((".deb", ".AppImage", ".sig", ".json", ".gz")):
        asset_url = asset["browser_download_url"]
        break

# Fallback: look for any asset that is just "koharu" (no extension)
if asset_url is None:
    for asset in release["assets"]:
        name = asset["name"]
        if name == "koharu" or name.startswith("koharu-") and "." not in name.split("-")[-1]:
            asset_url = asset["browser_download_url"]
            break

if asset_url is None:
    # Show available assets so user can manually pick
    print("\\n⚠️  Could not auto-detect the Linux binary. Available assets:")
    for a in release["assets"]:
        print(f"  • {a['name']}  →  {a['browser_download_url']}")
    raise RuntimeError(
        "Set asset_url manually in this cell to the correct Linux binary URL, then re-run."
    )

print(f"Downloading: {asset_url}")
BIN_DIR = "/opt/koharu"
os.makedirs(BIN_DIR, exist_ok=True)
BIN_PATH = os.path.join(BIN_DIR, "koharu")

subprocess.run(["wget", "-q", "-O", BIN_PATH, asset_url], check=True)

# Make executable
os.chmod(BIN_PATH, os.stat(BIN_PATH).st_mode | stat.S_IEXEC)

# Quick sanity check
ver = subprocess.check_output([BIN_PATH, "--version"], text=True).strip()
print(f"✅ Koharu binary ready: {ver}")\
""")

CELL2 = code("""\
#@title 🧠 Cell 2 — Pre-download model weights (optional but saves time)
import subprocess

BIN = "/opt/koharu/koharu"

# The --download flag tells Koharu to fetch all runtime libs / CUDA packages
# and exit without starting the server.
print("Pre-downloading runtime packages & CUDA libs …")
subprocess.run([BIN, "--download"], check=True)
print("✅ Runtime packages ready.")\
""")

CELL3 = code("""\
#@title 🚀 Cell 3 — Start Koharu headless server (port 3000)
import subprocess, time, threading, sys

BIN   = "/opt/koharu/koharu"
PORT  = 3000
LOG   = "/tmp/koharu_server.log"

# Kill any previous instance
subprocess.run(["pkill", "-f", BIN], stderr=subprocess.DEVNULL)
time.sleep(1)

# Launch in background
logfile = open(LOG, "w")
proc = subprocess.Popen(
    [BIN, "--headless", "--port", str(PORT)],
    stdout=logfile, stderr=subprocess.STDOUT,
    env={**__import__("os").environ, "RUST_LOG": "info"}
)

# Stream log output in a thread so we can see startup
def _tail():
    import time
    with open(LOG, "r") as f:
        while proc.poll() is None:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
            else:
                time.sleep(0.3)
t = threading.Thread(target=_tail, daemon=True)
t.start()

# Wait for server to be ready
import urllib.request
for i in range(60):
    time.sleep(2)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/v1/meta")
        print(f"\\n✅ Koharu server is UP on port {PORT}  (pid {proc.pid})")
        break
    except Exception:
        pass
else:
    print("\\n❌ Server did not start in time. Check /tmp/koharu_server.log")
    sys.exit(1)\
""")

CELL4 = code("""\
#@title 🌐 Cell 4 — Start Cloudflare Tunnel → public URL
import subprocess, time, re, threading

PORT = 3000

# Download cloudflared
subprocess.run(
    ["wget", "-q",
     "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
     "-O", "/usr/local/bin/cloudflared"],
    check=True
)
subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

# Kill previous tunnel if any
subprocess.run(["pkill", "-f", "cloudflared"], stderr=subprocess.DEVNULL)
time.sleep(1)

# Start tunnel
CF_LOG = "/tmp/cloudflared.log"
cf_logfile = open(CF_LOG, "w")
cf_proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
    stdout=cf_logfile, stderr=subprocess.STDOUT
)

# Wait for the URL to appear in logs
tunnel_url = None
for i in range(30):
    time.sleep(2)
    try:
        with open(CF_LOG, "r") as f:
            log = f.read()
        m = re.search(r"(https://[a-z0-9-]+\\.trycloudflare\\.com)", log)
        if m:
            tunnel_url = m.group(1)
            break
    except Exception:
        pass

if tunnel_url:
    print()
    print("=" * 60)
    print("🌸 YOUR KOHARU BACKEND URL:")
    print()
    print(f"   {tunnel_url}")
    print()
    print("Paste this into your local Koharu GUI:")
    print("  Settings → Backend URL")
    print("=" * 60)
    print()
    print("This URL is valid as long as this Colab session is alive.")
else:
    print("❌ Tunnel did not start. Check /tmp/cloudflared.log")
    with open(CF_LOG) as f:
        print(f.read()[-2000:])\
""")

CELL5 = code("""\
#@title 🩺 Cell 5 — Health-check & keep-alive
import time, urllib.request

PORT = 3000
URL = f"http://127.0.0.1:{PORT}/api/v1/meta"

print("Health-check loop running (prints every 60s, keeps Colab awake) …")
print("Press the ⬛ stop button to end.\\n")

try:
    while True:
        try:
            with urllib.request.urlopen(URL, timeout=10) as r:
                data = r.read().decode()
            print(f"[OK]  {time.strftime('%H:%M:%S')}  {data.strip()}")
        except Exception as e:
            print(f"[ERR] {time.strftime('%H:%M:%S')}  {e}")
        time.sleep(60)
except KeyboardInterrupt:
    print("\\nStopped.")\
""")

# ── assemble notebook ─────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.11.0"},
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "cells": [TITLE, CELL0, CELL1, CELL2, CELL3, CELL4, CELL5]
}

out_path = os.path.join(os.path.dirname(__file__) or ".", "koharu_backend.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook written to: {out_path}")
