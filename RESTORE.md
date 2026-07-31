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
