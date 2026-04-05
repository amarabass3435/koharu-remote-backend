# Remote Backend Plan — Colab-Hosted Koharu Backend

## Goal

Run all Koharu ML models (detect, segment, OCR, font-detect, inpaint) on a Google Colab
GPU, while the local desktop keeps a fully working GUI and state. From the user's perspective
the app behaves exactly as if everything ran locally — you just click the same buttons.

## How It Works (Architecture Overview)

```
[Local PC — Windows]                    [Google Colab — GPU T4/A100]
────────────────────                    ──────────────────────────────
Koharu Tauri GUI  ──── HTTP/REST ────►  koharu binary  (headless)
 (unchanged)                              └─ all ML models loaded
                                          └─ same /api/v1/* endpoints
                                          └─ Cloudflare Tunnel (free *.trycloudflare.com URL)
```

1. Colab builds the Koharu binary with CUDA (pre-compiled release is preferred).
2. Colab installs `cloudflared` and starts a **Cloudflare Quick Tunnel** — no account,
   no auth token required.
3. The notebook prints the public `*.trycloudflare.com` HTTPS URL.
4. You paste that URL into the Koharu GUI's "Server URL" setting, and everything
   works exactly as normal — upload images, click detect/OCR/inpaint, get results.
5. Translation (MiniMax / LLM) stays configured via the provider API key setting;
   no change needed on that path.

## Plan Status

- [x] Architecture decided
- [x] Colab notebook written (`colab/koharu_backend.ipynb`)
- [ ] Smoke test: import → detect → OCR → inpaint → export
- [ ] Document final ngrok URL wiring in the GUI

---

## Work Items

### Colab Notebook (`colab/koharu_backend.ipynb`)

- [x] Cell 0 — Check GPU, install system deps (build tools, CUDA libs)
- [x] Cell 1 — Clone repo from GitHub (or upload zip)
- [x] Cell 2 — Build `koharu` binary with CUDA feature flag
- [x] Cell 3 — Download / pre-warm model weights from HuggingFace
- [x] Cell 4 — Start koharu headless server on port 3000
- [x] Cell 5 — Install `cloudflared` & launch Quick Tunnel, print public `*.trycloudflare.com` URL
- [x] Cell 6 — (Optional) keep-alive / health-check loop

### Local GUI Wiring

- [ ] Confirm Koharu GUI has a "Server URL" field (Settings → Backend URL).
  - If yes: paste the ngrok URL that Colab prints → done.
  - If no: we patch the config to add a `remote_url` override and forward calls
    (see Fallback approach below).

### Fallback: Local Proxy Adapter (if GUI has no URL setting)

If the GUI cannot be redirected by config alone, we add a thin local proxy:

- [ ] Add `[remote]` section to `config.toml`:
  ```toml
  [remote]
  enabled = true
  url = "https://xxxx.trycloudflare.com"
  ```
- [ ] In `koharu-app/src/config.rs`: add `RemoteConfig` struct with `enabled` + `url`.
- [ ] In `koharu-rpc/src/api.rs`: add a middleware that, when `remote.enabled`, forwards
      the HTTP request to `remote.url` and streams the response back, bypassing local engines.
- [ ] Map: detect / recognize / inpaint / pipeline endpoints → remote forwarding.
- [ ] Local-only endpoints (documents, blobs, render, translate) remain untouched.

### Reliability Controls

- [ ] Timeout: 120 s connect, 600 s read (large images take time on Colab cold start).
- [ ] Health check: `/api/v1/meta` — if it fails, surface a clear error in GUI.
- [ ] On/off toggle in config so you can switch back to local (CPU) at any time.

---

## Device Limitations (Context)

- AMD CPU/iGPU — no CUDA path locally.
- `gemm-f32` panic in ONNX Runtime confirms the local ML path is broken on this hardware.
- Local build toolchain has had native dep gaps across sessions.
- **Decision**: keep GUI and document state local; offload all ML inference to Colab GPU.

---

## Notebook Location

`colab/koharu_backend.ipynb` — ready to open in Google Colab.

Upload it at: https://colab.research.google.com → File → Open notebook → Upload.

> **Why Cloudflare Tunnel instead of ngrok?**
> - No account or auth token required.
> - No rate limits on the free tier.
> - Stable HTTPS URL valid for the entire Colab session.
> - `cloudflared` is a single static binary — trivial to install in Colab.
