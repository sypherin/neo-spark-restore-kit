const PORT = +(process.env.GW_PORT ?? 8002);
const RATE = +(process.env.GW_RATE_LIMIT ?? 60);           // req/min per IP
const MAX_BODY = +(process.env.GW_MAX_BODY ?? 2*1024*1024);
const KEYS_FILE = process.env.GW_KEYS_FILE ?? `${process.env.HOME}/.config/llm-gateway/keys.json`;
// ONE endpoint, three models — routed by path prefix, all behind the same key:
const ROUTES = [
  { prefix: "/vlm/", target: "http://127.0.0.1:8080", strip: "/vlm" },   // Qwen3-VL vision
  { prefix: "/layout", target: "http://127.0.0.1:8090", strip: ""     }, // Surya OCR (DocFlow POSTs /layout)
  { prefix: "/",     target: "http://127.0.0.1:8001", strip: ""     },   // Gemma LLM (default)
];
const keys = new Set((JSON.parse(await Bun.file(KEYS_FILE).text())).map((k:any)=>k.key));
const hits = new Map<string, number[]>();
Bun.serve({ port: PORT, hostname: "127.0.0.1", async fetch(req) {
  const url = new URL(req.url);
  if (url.pathname === "/healthz") return Response.json({ ok: true });
  const key = (req.headers.get("authorization") ?? "").replace(/^Bearer /, "");
  if (!keys.has(key)) return new Response("unauthorized", { status: 401 });
  const ip = req.headers.get("cf-connecting-ip") ?? "local";
  const now = Date.now(), win = (hits.get(ip) ?? []).filter(t => now - t < 60000);
  if (win.length >= RATE) return new Response("rate limited", { status: 429 });
  win.push(now); hits.set(ip, win);
  const body = req.method === "POST" ? await req.arrayBuffer() : undefined;
  if (body && body.byteLength > MAX_BODY) return new Response("too large", { status: 413 });
  const r = ROUTES.find(x => url.pathname.startsWith(x.prefix))!;
  const fwd = r.target + url.pathname.slice(r.strip.length) + url.search;
  return fetch(fwd, { method: req.method, headers: { "content-type": "application/json" }, body });
}});
console.log(`gateway :${PORT} → Gemma :8001 · Qwen-VL :8080 · Surya :8090`);
