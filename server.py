"""
Surya layout + OCR HTTP server — document-pipeline OCR service.

Exposes:

    POST /layout
        Content-Type: multipart/form-data
        Form field: image  (the page image bytes)
    Returns JSON:
        {
          "tokens": [
            { "type": "Title|Text|Picture|Table|TableCell|Caption|...",
              "bbox": [xmin, ymin, xmax, ymax],     # normalised 0-1
              "text": "actual OCR'd text",          # filled by RecognitionPredictor
              "conf": 0.0-1.0
            },
            ...
          ],
          "modelUsed": "surya-layout+det+rec",
          "pageCount": 1,
          "tookMs": <int>
        }

Pipeline (full nemotron-parse equivalent, all-local):
    1. DetectionPredictor   — finds text bounding boxes
    2. RecognitionPredictor — OCRs the text inside each detected bbox
    3. LayoutPredictor      — classifies each region (Title / Text / Picture / TableCell / etc.)
    Output: each token has bbox + type + actual text + confidence.

Two foundation predictors are used because Surya ships separate checkpoints:
  - FoundationPredictor(LAYOUT_MODEL_CHECKPOINT)        for region classification
  - FoundationPredictor(FOUNDATION_MODEL_CHECKPOINT)    for OCR text recognition
"""

import io
import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [surya] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Lazy-loaded predictors. First /layout request triggers all model downloads
# (~3-4 GB total: detection + recognition + layout + foundation backbones).
_pipeline = None


def get_pipeline():
    """
    Returns a tuple (detector, recognizer, layout) of Surya predictors. Lazy-
    initialised on first request so the service comes up immediately and the
    download happens at /layout time.

    SURYA_DEVICE env: 'cuda' (= ROCm on AMD via PyTorch's CUDA-compat layer) or
    'cpu'. Default 'cuda' if torch.cuda.is_available(), else 'cpu'.
    """
    global _pipeline
    if _pipeline is None:
        import os
        import torch
        env_device = os.environ.get("SURYA_DEVICE")
        if env_device:
            device = env_device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("first request — loading Surya pipeline (det + rec + layout) on device=%s — may download 3-4 GB on cold start", device)

        from surya.foundation import FoundationPredictor
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.layout import LayoutPredictor
        from surya.settings import settings as surya_settings

        # Two foundation predictors, different checkpoints — layout vs text.
        foundation_layout = FoundationPredictor(checkpoint=surya_settings.LAYOUT_MODEL_CHECKPOINT, device=device)
        foundation_text = FoundationPredictor(checkpoint=surya_settings.FOUNDATION_MODEL_CHECKPOINT, device=device)

        detector = DetectionPredictor(device=device)
        recognizer = RecognitionPredictor(foundation_text)
        layout = LayoutPredictor(foundation_layout)

        _pipeline = (detector, recognizer, layout)
        log.info("Surya pipeline loaded on %s", device)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("surya HTTP server starting on :8090")
    yield
    log.info("surya HTTP server shutting down")


app = FastAPI(title="surya layout+OCR", lifespan=lifespan)


def _health_payload():
    return {
        "ok": True,
        "modelLoaded": _pipeline is not None,
        "device": os.environ.get("SURYA_DEVICE", "cuda"),
    }


@app.get("/healthz")
def healthz():
    return _health_payload()


@app.get("/health")
def health():
    # Alias of /healthz so callers using either path get a 200.
    return _health_payload()


def _polygon_to_bbox(poly):
    """Normalise Surya bbox/polygon to (xmin, ymin, xmax, ymax) ints."""
    if not poly:
        return None
    if isinstance(poly[0], (list, tuple)):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    if len(poly) >= 4:
        return (poly[0], poly[1], poly[2], poly[3])
    return None


def _box_inside(inner, outer):
    """True if `inner`'s centroid is inside `outer`. Used to associate a text
    box with the layout region it sits in."""
    ix1, iy1, ix2, iy2 = inner
    cx = (ix1 + ix2) / 2
    cy = (iy1 + iy2) / 2
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


# ---------------------------------------------------------------------------
# Surya 2 backend (2026-06-06).
#
# SURYA_BACKEND=v2 routes /layout through the Surya 2 650M VLM served by
# llama-server (llama-surya2.service, :8093) instead of the v1 torch pipeline.
# Why: v1 rec stage is autoregressive-per-line on a bandwidth-bound iGPU
# (4 min on dense phone photos) and ROCm JIT-compiles kernels per shape;
# Surya 2 on llama.cpp does the whole page in ONE pass with no JIT cliffs
# (measured 23.3s vs 246.7s on the same 1807x2400 doc, 2026-06-06).
#
# The v1 pipeline stays loaded and is the AUTOMATIC per-request fallback:
# any v2 error, truncation, or decoder repeat-loop falls through to v1.
# Rollback = SURYA_BACKEND=v1 (the default) + container restart.
# ---------------------------------------------------------------------------
import base64
import html as _html
import json as _json
import re as _re
import urllib.request as _urlreq

SURYA_BACKEND = os.environ.get("SURYA_BACKEND", "v1").strip().lower()
# Hot-flip override: if this file exists its content ("v1"/"v2") wins over the
# env var, checked per request — so switching backends is
#   echo v2 > ~/surya/backend.flag
# with NO container restart, and rollback is echo v1 (or rm the file).
SURYA_BACKEND_FILE = os.environ.get("SURYA_BACKEND_FILE", os.path.expanduser("~/surya/backend.flag"))


def _active_backend():
    try:
        with open(SURYA_BACKEND_FILE) as f:
            v = f.read().strip().lower()
        if v in ("v1", "v2"):
            return v
    except OSError:
        pass
    return SURYA_BACKEND


SURYA2_URL = os.environ.get("SURYA2_URL", "http://127.0.0.1:8093")
SURYA2_TIMEOUT_S = int(os.environ.get("SURYA2_TIMEOUT_S", "300"))
# 24576 (was 12288, was 4096): the 51-page bulk scan (2026-06-06) had 163-line
# grid pages whose LEGITIMATE output exceeds 12k — not loops. Model n_ctx_train
# is 262144, serving context now -c 32768; image+prompt ~2.5-4.2k so 24k output
# fits. Loop pages are still caught by the repeat-loop guard (pattern-based,
# not cap-based). Bulk multi-page docs remain routed away regardless (time).
SURYA2_MAX_TOKENS = int(os.environ.get("SURYA2_MAX_TOKENS", "24576"))

# repeat_penalty stops the VLM decoder-loop on sparse ruled-table scans (Canon
# fax/CCITTFax invoices etc.) where the model reads the page fine but then emits
# empty <div> rows to the token cap → the loop guard throws the whole read away
# → slow v1 fallback → DLQ. 1.15 tested 2026-06-09 on the SAN HUP $6 invoice:
# clean finish=stop, 31 divs, ALL fields (vs finish=length / looped at 1.0).
# Mild enough not to garble legit repetitive DO tables the way DRY 0.8 does.
# Env-tunable; set SURYA2_REPEAT_PENALTY=1.0 to disable.
SURYA2_REPEAT_PENALTY = float(os.environ.get("SURYA2_REPEAT_PENALTY", "1.15"))

# --- Surya 1 (v1 torch) activation alert ----------------------------------
# Zach 2026-06-09: "I do NOT want to activate Surya 1; alert me if it ever is."
# v1 is the slow (~6 min) torch fallback being retired. If Surya 2 fails ALL its
# passes (plain → repeat_penalty → DRY) and v1 kicks in, ping Zach so the silent
# slow-path is never invisible. Best-effort + throttled; NEVER breaks OCR.
_V1_ALERT_CHAT = os.environ.get("SURYA_ALERT_CHAT_ID", "")  # operator opt-in via env
_last_v1_alert = [0.0]

def _read_tg_token():
    try:
        t = open(os.path.expanduser("~/.config/ocr-watchdog/telegram-bot-token")).read().strip()
        if t:
            return t
    except OSError:
        pass
    try:
        for line in open(os.path.expanduser("~/.env.local.lyra-backend")):
            if line.startswith("LYRA_TG_BOT_TOKEN"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None

def _alert_v1_activated(reason):
    """Best-effort, throttled Telegram alert when the v1 torch fallback runs."""
    import time as _t
    try:
        if _t.time() - _last_v1_alert[0] < 600:   # max 1 alert / 10 min (anti-spam)
            return
        tok = _read_tg_token()
        if not tok:
            return
        _last_v1_alert[0] = _t.time()
        msg = ("⚠️ SURYA 1 (v1 torch) ACTIVATED on document OCR — Surya 2 "
               "failed all passes (plain→repeat_penalty→DRY).\nReason: " + str(reason)[:170] +
               "\nThis is the slow path; the doc may DLQ. Check surya-only logs / re-test the doc.")
        data = _json.dumps({"chat_id": _V1_ALERT_CHAT, "text": msg}).encode()
        req = _urlreq.Request("https://api.telegram.org/bot%s/sendMessage" % tok,
                              data=data, headers={"Content-Type": "application/json"})
        _urlreq.urlopen(req, timeout=8)
    except Exception:
        pass

# Training-time contract prompt from surya/inference/prompts.py — do NOT
# paraphrase (the exact wording is what the model was trained on).
_SURYA2_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)
# Canonical per-block prompt (same contract file) — used by the letterhead
# logo pass below.
_SURYA2_BLOCK_PROMPT = "OCR this block image to HTML."

# Letterhead logo pass (2026-06-06, Zach-requested): Surya 2 is TRAINED to
# skip Image/Figure blocks in the full-page pass (bare <img/>, instruction
# can't override — tested), so logo-only supplier names never reach the
# structurer (GEISLER packing list lost its supplier; Qwen-VL recovery only
# rescues probabilistically). Fix: crop image-blocks in the letterhead region
# and run them through the canonical BLOCK prompt; returned text rides on the
# token so Gemma sees it. Bounded: top of page only, max N crops, short
# timeout, failures leave the token as-is.
SURYA2_LOGO_PASS = os.environ.get("SURYA2_LOGO_PASS", "1") not in ("0", "false")
# Zach 2026-06-06: read ALL image blocks unconditionally ("it's not a matter
# of bad [fields]") — stamps/chops/footer marks carry receivedBy + company
# names, not just letterhead logos. Region default = whole page.
SURYA2_LOGO_MAX_CROPS = int(os.environ.get("SURYA2_LOGO_MAX_CROPS", "4"))
SURYA2_LOGO_REGION = float(os.environ.get("SURYA2_LOGO_REGION", "1.0"))  # top fraction of page

# Surya 2 layout labels -> v1 token types the DocFlow pipeline already knows.
_SURYA2_LABEL_MAP = {
    "Section-Header": "Section-header",
    "Page-Header": "Page-header",
    "Page-Footer": "Page-footer",
    "Image": "Picture",
    "Equation-Block": "Formula",
    "List-Group": "List-item",
    "Table-Of-Contents": "Text",
    "Complex-Block": "Text",
    "Code-Block": "Text",
    "Chemical-Block": "Text",
    "Diagram": "Figure",
    "Bibliography": "Text",
    "Form": "Text",
    # Caption / Footnote / Table / Text / Figure pass through unchanged.
}

# v2 emits no per-line confidence; downstream field-confidence comes from the
# extraction stage, but the token schema requires a value.
_SURYA2_CONF = float(os.environ.get("SURYA2_CONF", "0.95"))


def _detect_repeat_loop(text, base_max_repeats=4, window_size=500, scaling_factor=3.0):
    """True iff the tail of `text` ends in a repeating sequence — the typical
    VLM decoder failure mode (same div/phrase until max_tokens). Ported from
    surya v2 / chandra detect_repeat_token."""
    if not text:
        return False
    for seq_len in range(1, window_size // 2 + 1):
        candidate = text[-seq_len:]
        max_repeats = int(base_max_repeats * (1 + scaling_factor / seq_len))
        repeats = 0
        pos = len(text) - seq_len
        while pos >= 0 and text[pos: pos + seq_len] == candidate:
            repeats += 1
            pos -= seq_len
        if repeats > max_repeats:
            return True
    return False


_TABLE_RE = _re.compile(r"<table[^>]*>.*?</table>", _re.S | _re.I)
_TR_RE = _re.compile(r"<tr[^>]*>(.*?)</tr>", _re.S | _re.I)
_CELL_RE = _re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", _re.S | _re.I)
_TAG_RE = _re.compile(r"<[^>]+>")
_DIV_RE = _re.compile(r"<div\b([^>]*)>(.*?)</div>", _re.S | _re.I)
_ATTR_BBOX_RE = _re.compile(r'data-bbox="(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"')
_ATTR_LABEL_RE = _re.compile(r'data-label="([^"]+)"')


def _html_to_text(fragment):
    """Strip an HTML fragment to plain text. <br>/<p> boundaries -> newlines,
    tables -> one line per row with ' | ' between cells, entities unescaped."""
    def table_to_text(m):
        rows = []
        for tr in _TR_RE.findall(m.group(0)):
            cells = [_TAG_RE.sub("", c).strip() for c in _CELL_RE.findall(tr)]
            if any(cells):
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    s = _TABLE_RE.sub(table_to_text, fragment)
    s = _re.sub(r"<br\s*/?>", "\n", s, flags=_re.I)
    s = _re.sub(r"</p>", "\n", s, flags=_re.I)
    s = _TAG_RE.sub("", s)
    s = _html.unescape(s)
    return "\n".join(line.strip() for line in s.split("\n") if line.strip())


def _surya2_ocr(img_bytes, content_type, dry=False, penalty=False, page_size=None):
    """One-pass full-page OCR via the Surya 2 VLM. Returns a token list in the
    exact v1 schema, or raises (caller retries, then falls back to v1).

    penalty=True applies repeat_penalty — RETRY-ONLY rescue for the empty-form
    decoder loop (Canon/CCITTFax invoices: a couple real rows then a big blank
    ruled table → the model spams <tr><td></td>…</tr> to the token cap). Kept
    OFF the clean first pass on purpose (2026-06-09, Zach): a global penalty
    risks distorting legitimately repetitive DO line-item tables. Validated on
    the SAN HUP invoice: clean finish=stop / all fields vs finish=length looped.

    dry=True enables llama.cpp's DRY sampler — a SECOND rescue retry. NOT default
    (2026-06-06: default DRY corrupts repetitive DO tables, "S45C…"→"blackeeniig"),
    but as a rescue it reads handwriting the v1 fallback missed (DET 140 p2)."""
    mime = content_type if content_type in ("image/jpeg", "image/png", "image/webp") else "image/jpeg"
    payload = {
        "temperature": 0,
        "max_tokens": SURYA2_MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"}},
                {"type": "text", "text": _SURYA2_PROMPT},
            ],
        }],
    }
    if penalty:
        # retry-only — breaks the empty-form-row decoder loop without touching
        # the clean first pass (which 95% of docs use). SURYA2_REPEAT_PENALTY=1.0 disables.
        payload["repeat_penalty"] = SURYA2_REPEAT_PENALTY
    if dry:
        payload["dry_multiplier"] = float(os.environ.get("SURYA2_DRY_MULTIPLIER", "0.8"))
        payload["dry_allowed_length"] = int(os.environ.get("SURYA2_DRY_ALLOWED_LENGTH", "4"))
    req = _urlreq.Request(
        f"{SURYA2_URL}/v1/chat/completions",
        data=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _urlreq.urlopen(req, timeout=SURYA2_TIMEOUT_S) as resp:
        body = _json.loads(resp.read())

    choice = body["choices"][0]
    out = choice["message"]["content"] or ""
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"surya2 output truncated at {SURYA2_MAX_TOKENS} tokens")
    if _detect_repeat_loop(out):
        raise RuntimeError("surya2 decoder repeat-loop detected")

    # Phase 1 — collect raw blocks.
    blocks = []
    for attrs, inner in _DIV_RE.findall(out):
        mb = _ATTR_BBOX_RE.search(attrs)
        ml = _ATTR_LABEL_RE.search(attrs)
        if not mb:
            continue
        x0, y0, x1, y1 = (int(v) for v in mb.groups())
        label = ml.group(1) if ml else "Text"
        if label == "Blank-Page":
            continue
        ttype = _SURYA2_LABEL_MAP.get(label, label)
        text = _html_to_text(inner)
        if not text and ttype not in ("Picture", "Figure"):
            continue
        blocks.append((ttype, text, x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0))

    # Phase 2 — orientation vote on RAW text blocks (pixel space). A genuinely
    # sideways scan has predominantly TALL text blocks. In that case we must
    # NOT line-split: the synthetic horizontal strips would mask the pipeline's
    # sideways detector (horizFrac), so it never rotates + re-OCRs and the
    # viewer shows the page lying on its side (Boneham, 2026-06-06). Returning
    # block-level tokens lets the existing rotate+re-OCR machinery fire; the
    # second (upright) pass then line-splits normally.
    split_lines = True
    if page_size:
        W, H = page_size
        votes = [(x1 - x0) * W >= (y1 - y0) * H
                 for ttype, text, x0, y0, x1, y1 in blocks
                 if ttype not in ("Picture", "Figure", "Table") and len(text.strip()) >= 2]
        if len(votes) >= 4 and sum(votes) / len(votes) < 0.4:
            split_lines = False
            log.info("surya2: raw blocks predominantly tall (%d/%d wide) — page likely sideways; "
                     "emitting block-level tokens so the pipeline's rotation pass can fire",
                     sum(votes), len(votes))

    # Phase 3 — emit tokens (line-split only for upright pages). Line strips
    # restore v1-like granularity for the pipeline's heuristics + replica view.
    tokens = []
    for ttype, text, bx0, by0, bx1, by1 in blocks:
        if split_lines:
            lines = [ln for ln in text.split("\n") if ln.strip()] or [text]
        else:
            lines = [text.replace("\n", " ")]
        n = len(lines)
        for i, ln in enumerate(lines):
            tokens.append({
                "type": ttype,
                "bbox": [bx0, by0 + (by1 - by0) * i / n, bx1, by0 + (by1 - by0) * (i + 1) / n],
                "text": ln,
                "conf": _SURYA2_CONF,
            })
    if not tokens:
        raise RuntimeError("surya2 returned no parseable blocks")
    return tokens


def _surya2_logo_pass(tokens, pil_img):
    """Crop letterhead-region image blocks and OCR them via the canonical
    BLOCK prompt. Mutates matching tokens' text in place. Never raises."""
    if not SURYA2_LOGO_PASS or pil_img is None:
        return tokens
    W, H = pil_img.size
    done = 0
    for t in tokens:
        if done >= SURYA2_LOGO_MAX_CROPS:
            break
        if t.get("type") not in ("Picture", "Figure") or t.get("text"):
            continue
        x0, y0, x1, y1 = t["bbox"]
        if y0 > SURYA2_LOGO_REGION:          # letterhead region only
            continue
        if (x1 - x0) * (y1 - y0) < 0.005:    # skip tiny marks/ticks
            continue
        try:
            pad = 6
            crop = pil_img.crop((max(0, int(x0 * W) - pad), max(0, int(y0 * H) - pad),
                                 min(W, int(x1 * W) + pad), min(H, int(y1 * H) + pad)))
            buf = io.BytesIO()
            crop.convert("RGB").save(buf, format="JPEG", quality=92)
            payload = {
                "temperature": 0,
                "max_tokens": 512,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                        {"type": "text", "text": _SURYA2_BLOCK_PROMPT},
                    ],
                }],
            }
            payload["logprobs"] = True
            payload["top_logprobs"] = 1
            req = _urlreq.Request(f"{SURYA2_URL}/v1/chat/completions",
                                  data=_json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=60) as resp:
                body = _json.loads(resp.read())
            choice = body["choices"][0]
            raw = choice["message"]["content"] or ""
            text = _html_to_text(raw)
            # ── acceptance gates (2026-06-06, Zach: "only high-confidence
            # letters"). Crops of non-text marks (QR codes, decorative strips)
            # made the model hallucinate junk ("() () ()", "Carrollo"×30,
            # Sanskrit, a confident "GE" from a QR). Calibrated on tonight's
            # real crops: genuine logo text decodes at mean p≈0.90 with ~7%
            # sub-0.5 tokens; garbage runs 0.69-0.83 with 17-28%.
            import math
            lps = [tk.get("logprob", 0.0) for tk in (choice.get("logprobs") or {}).get("content", [])]
            probs = [math.exp(l) for l in lps] if lps else []
            mean_p = (sum(probs) / len(probs)) if probs else 0.0
            frac_low = (sum(1 for p in probs if p < 0.5) / len(probs)) if probs else 1.0
            reject = None
            if not text:
                reject = "empty"
            elif _detect_repeat_loop(text):
                reject = "repeat-loop"
            elif len(text) > 120:
                reject = f"too-long({len(text)})"
            elif sum(c.isalnum() for c in text) < 2:
                reject = "no-alnum"
            else:
                # Two-tier confidence (2026-06-06): STRICT for letterhead-region
                # fills (supplier identity — a misread is silent and harmful);
                # RELAXED for stamp-shaped content — chops mix crisp print with
                # handwritten dates, so frac_low runs 15-30% on REAL stamps
                # (observed: a genuine chop died at mean=0.84/low=0.20). A
                # relaxed fill must ALSO look like stamp data (GRN/RECEIVED/
                # company/date patterns), so QR-hallucinations ("GE" at
                # 0.83/0.17) still die in the strict tier.
                stamp_shaped = bool(_re.search(
                    r"(GRN\s?\d|GOODS\s+RECEI|RECEIVED|CHONG\s*FONG|\d{1,2}\s+[A-Z]{3,9}\s+20\d{2}|SUBJECT\s+TO\s+CONFIRMATION)",
                    text, _re.I))
                if stamp_shaped:
                    if mean_p < 0.70 or frac_low > 0.35:
                        reject = f"low-conf-stamp(mean={mean_p:.2f},low={frac_low:.2f})"
                elif mean_p < 0.85 or frac_low > 0.12:
                    reject = f"low-conf(mean={mean_p:.2f},low={frac_low:.2f})"
            if reject:
                log.info("logo-pass: DISCARDED %s block bbox=%s (%s)", t["type"], [round(v, 2) for v in t["bbox"]], reject)
            else:
                t["text"] = text
                done += 1
                log.info("logo-pass: filled %s block bbox=%s with %d chars (mean_p=%.2f)", t["type"], [round(v, 2) for v in t["bbox"]], len(text), mean_p)
        except Exception as e:
            log.warning("logo-pass crop failed (%s) — token left as-is", e)
    return tokens


@app.post("/layout")
def layout(image: UploadFile = File(...)):
    # SYNC handler (2026-06-06): was `async def`, but the v2 path makes a
    # BLOCKING urllib call (up to SURYA2_TIMEOUT_S) inside it, which froze the
    # event loop → /healthz starved for the whole OCR → the ops watchdog saw
    # "6 healthz fails + CPU idle" (GPU work lives in the llama-surya2 process,
    # so this container looked dead) and restarted us mid-batch. A sync def
    # runs in Starlette's threadpool, keeping the loop + healthz responsive.
    t0 = time.time()
    try:
        img_bytes = image.file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image decode failed: {e}")

    # --- Surya 2 path (SURYA_BACKEND=v2 or backend.flag): one-pass VLM OCR via
    # llama-server. Any failure falls through to the v1 torch pipeline below.
    if _active_backend() == "v2":
        tokens = None
        model_used = "surya2-vlm"
        try:
            tokens = _surya2_ocr(img_bytes, image.content_type or "", page_size=img.size)
        except Exception as e:
            # Truncation / repeat-loop → rescue RETRY 1: repeat_penalty (breaks
            # the empty-form-row loop; retry-only so the clean first pass + 95%
            # of docs are untouched). Then RETRY 2: DRY sampler. Then v1 torch.
            log.warning("surya2 attempt 1 failed (%s) — retry with repeat_penalty", e)
            try:
                tokens = _surya2_ocr(img_bytes, image.content_type or "", penalty=True, page_size=img.size)
                model_used = "surya2-vlm-rp"
            except Exception as e2:
                log.warning("surya2 repeat_penalty retry failed (%s) — retry with DRY sampler", e2)
                try:
                    tokens = _surya2_ocr(img_bytes, image.content_type or "", dry=True, page_size=img.size)
                    model_used = "surya2-vlm-dry"
                except Exception as e3:
                    log.warning("surya2 DRY retry failed (%s) — falling back to v1 torch pipeline", e3)
                    _alert_v1_activated(e3)
        if tokens:
            tokens = _surya2_logo_pass(tokens, img)
            took_ms = int((time.time() - t0) * 1000)
            log.info("surya2 ok page=%dx%d tokens=%d took=%dms model=%s",
                     img.width, img.height, len(tokens), took_ms, model_used)
            return JSONResponse({
                "tokens": tokens,
                "modelUsed": model_used,
                "pageCount": 1,
                "tookMs": took_ms,
            })

    try:
        detector, recognizer, layout_predictor = get_pipeline()

        # Per-step timing — added 2026-05-28 to locate the bottleneck.
        # The earlier `took=Xms` log only shows total time; for a 1350x1800
        # page with 70 text_lines we see ~160s total but didn't know if it
        # was detection, OCR-per-line, or layout. With batch_size=1 the OCR
        # step processes one line at a time → expected dominant cost.
        t_det0 = time.time()
        det_results = detector([img])
        t_det = time.time() - t_det0

        # Step 2: OCR text in each detected bbox.
        # 2026-05-28 step-timing investigation found rec stage is 99% of
        # total time (171s of 172s on a 70-line 1800x2400 page). Batch-size
        # knobs (det=32, rec=64) AND math_mode=False made no measurable
        # difference — confirms the bottleneck is autoregressive decoder
        # throughput on Strix Halo iGPU, not batching or grammar. Real fixes
        # require either a smaller recognizer model, a different OCR
        # pipeline, or hardware with more memory bandwidth. Kept batch hints
        # for future-friendliness (newer Surya versions may use them better).
        t_rec0 = time.time()
        try:
            rec_results = recognizer(
                [img], det_predictor=detector,
                detection_batch_size=32, recognition_batch_size=32,
            )
        except TypeError:
            try:
                rec_results = recognizer([img], det_predictor=detector, detection_batch_size=32)
            except TypeError:
                rec_results = recognizer([img])
        t_rec = time.time() - t_rec0

        # Step 3: classify regions
        t_lay0 = time.time()
        layout_results = layout_predictor([img])
        t_lay = time.time() - t_lay0
        log.info("surya step timing — det=%.1fs rec=%.1fs lay=%.1fs", t_det, t_rec, t_lay)
    except Exception as e:
        log.exception("surya pipeline failed")
        raise HTTPException(status_code=500, detail=f"surya pipeline failed: {e}")

    page_w = max(img.width, 1)
    page_h = max(img.height, 1)

    # Pull recognized text bboxes (each has bbox + text + confidence)
    rec_lines = []
    if rec_results and len(rec_results) > 0:
        page = rec_results[0]
        for line in getattr(page, "text_lines", []) or []:
            poly = getattr(line, "polygon", None) or getattr(line, "bbox", None)
            bb = _polygon_to_bbox(poly)
            if not bb:
                continue
            rec_lines.append({
                "bbox": bb,
                "text": (getattr(line, "text", "") or "").strip(),
                "conf": float(getattr(line, "confidence", 0.0) or 0.0),
            })

    # Pull layout regions (each has bbox + label/type + confidence)
    layout_regions = []
    if layout_results and len(layout_results) > 0:
        page = layout_results[0]
        for b in getattr(page, "bboxes", []) or []:
            poly = getattr(b, "polygon", None) or getattr(b, "bbox", None)
            bb = _polygon_to_bbox(poly)
            if not bb:
                continue
            layout_regions.append({
                "bbox": bb,
                "type": getattr(b, "label", None) or "Text",
                "conf": float(getattr(b, "confidence", 0.0) or 0.0),
            })

    # Strategy: emit one token per RECOGNISED text line. Type assigned from
    # whichever layout region the text-line centroid falls inside; default
    # "Text" if no enclosing region. This gives nemotron-parse-equivalent
    # output: every token has bbox + type + actual OCR text.
    tokens = []
    for line in rec_lines:
        ibb = line["bbox"]
        ttype = "Text"
        for r in layout_regions:
            if _box_inside(ibb, r["bbox"]):
                ttype = r["type"]
                break
        x1, y1, x2, y2 = ibb
        tokens.append({
            "type": ttype,
            "bbox": [x1 / page_w, y1 / page_h, x2 / page_w, y2 / page_h],
            "text": line["text"],
            "conf": line["conf"],
        })

    # Also emit "Picture" / "Figure" regions that have NO text (logos, photos)
    # — important for the pipeline UI's replica view, AND so the structurer
    # can see "there's a picture/letterhead region we couldn't OCR".
    for r in layout_regions:
        if r["type"] not in ("Picture", "Figure"):
            continue
        # Skip if any text line already lives inside this region (already covered)
        rb = r["bbox"]
        if any(_box_inside(line["bbox"], rb) for line in rec_lines):
            continue
        x1, y1, x2, y2 = rb
        tokens.append({
            "type": r["type"],
            "bbox": [x1 / page_w, y1 / page_h, x2 / page_w, y2 / page_h],
            "text": "",
            "conf": r["conf"],
        })

    took_ms = int((time.time() - t0) * 1000)
    log.info("layout+ocr ok page=%dx%d text_lines=%d layout_regions=%d tokens=%d took=%dms",
             img.width, img.height, len(rec_lines), len(layout_regions), len(tokens), took_ms)
    return JSONResponse({
        "tokens": tokens,
        "modelUsed": "surya-layout+det+rec",
        "pageCount": 1,
        "tookMs": took_ms,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
