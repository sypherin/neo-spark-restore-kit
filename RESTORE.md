# Surya OCR restore — DGX Spark (`~/surya` wiped, surya.service crash-looping)

Matches the "Private LLM on DGX Spark — Setup Guide", §6. The venv
(`~/surya-venv`) is a plain pip install and is almost certainly intact — the only thing
missing is the `server.py` wrapper, which does NOT ship with the upstream surya package.

## Restore (normal case — venv intact)

```bash
mkdir -p ~/surya
# put server.py from THIS repo at ~/surya/server.py
systemctl --user restart surya.service
curl -s localhost:8090/healthz        # {"ok":true,...} = recovered
```

That's it. Do not reinstall anything first — per guide §5's own advice: diagnose, don't reinstall.

## If the venv is ALSO broken (only then)

```bash
python3 -m venv ~/surya-venv
~/surya-venv/bin/pip install surya-ocr fastapi uvicorn pillow python-multipart
```
(guide §6a — v1 torch models auto-download from Hugging Face on the first /layout call, ~1-2 GB)

## Surya 2 GGUF — no rebuild needed, ever

The v2 accelerator weights are a PUBLIC download (guide §6b) — public Hugging Face repo, not something built on the box:

```bash
~/surya-venv/bin/hf download datalab-to/surya-ocr-2-gguf \
  surya-2.gguf surya-2-mmproj.gguf --local-dir ~/models/surya-2
```

If `~/models/surya-2/` still exists on the box, nothing to do. `llama-surya2.service` (:8093)
serves them with the already-built llama-server binary. Start order: `llama-surya2` first,
then `surya` — the unit ordering (`After=llama-surya2.service`) handles this on boot.

## How the tiers work

`server.py` = the /layout FastAPI wrapper (`uvicorn server:app`, :8090). `SURYA_BACKEND=v2`
routes through the fast 650M GGUF on :8093; any decoder-loop/error auto-falls back to the
v1 torch pipeline and logs it. Force v1: `echo v1 > ~/surya/backend.flag` (no restart).

## Alternative pull, on the box itself via Deneb

```bash
TOK=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/deneb/config.json')))['token'])")
curl -fsSL -H "Authorization: Bearer $TOK" https://deneb-engine.altronis.sg/artifact/surya-server.py -o ~/surya/server.py
```

## Stack guard — survive reboots and crashes

`spark-stack-guard.sh` (+ `.service` + `.timer`) checks every 2 minutes that all five
services are enabled, active, and answering their health endpoints; restarts anything
dead; enables anything that would not survive a reboot; enables linger. Slow model
loads get a 10-minute warmup grace before an unresponsive-restart. Install:

```bash
mkdir -p ~/bin
curl -fsSL https://raw.githubusercontent.com/sypherin/neo-spark-restore-kit/master/spark-stack-guard.sh -o ~/bin/spark-stack-guard.sh
chmod +x ~/bin/spark-stack-guard.sh
curl -fsSL https://raw.githubusercontent.com/sypherin/neo-spark-restore-kit/master/spark-stack-guard.service -o ~/.config/systemd/user/spark-stack-guard.service
curl -fsSL https://raw.githubusercontent.com/sypherin/neo-spark-restore-kit/master/spark-stack-guard.timer -o ~/.config/systemd/user/spark-stack-guard.timer
systemctl --user daemon-reload
systemctl --user enable --now spark-stack-guard.timer
~/bin/spark-stack-guard.sh          # run once now; then: journalctl --user -t spark-guard -n 20
```

Reboot survival checklist the guard enforces continuously: linger enabled, all five
units `enabled`, everything restarted on failure (units already carry Restart=always/
on-failure for crashes; the guard covers the stayed-down and never-enabled cases).

## Embeddings service + full-stack health (the two routes the simple gateway lacked)

**`/upstreams/health`** — an authed endpoint on the gateway that probes every backend
(gemma, vlm, surya, surya2, embed) and returns one JSON scorecard. Apps and monitors poll
this ONE url instead of poking model ports. It ships in this repo's `gateway.ts` — just
re-pull it and restart `llm-gateway`.

**`/embed/v1`** — needs a small embeddings model (bge-m3, same as the reference stack),
served by the llama-server binary already built on the box:

```bash
~/surya-venv/bin/hf download gpustack/bge-m3-GGUF bge-m3-FP16.gguf --local-dir ~/models/bge-m3
```

`~/.config/systemd/user/llama-embed.service`:

```ini
[Unit]
Description=bge-m3 embeddings (CUDA - DGX Spark)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=%h/llama-cpp-turboquant/build/bin/llama-server \
  -m %h/models/bge-m3/bge-m3-FP16.gguf \
  --embedding --pooling cls -ngl 99 -c 8192 \
  --host 127.0.0.1 --port 8091
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now llama-embed
curl -s localhost:8091/health
# then re-pull gateway.ts (adds the /embed route + /upstreams/health) and restart:
curl -fsSL https://raw.githubusercontent.com/sypherin/neo-spark-restore-kit/master/gateway.ts -o ~/llm-gateway/gateway.ts
systemctl --user restart llm-gateway
```

The stack guard watches llama-embed out of the box (re-pull spark-stack-guard.sh if you
installed an earlier version). bge-m3 FP16 is ~1.2 GB — negligible next to the LLMs.
