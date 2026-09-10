// Songgot Pocket: the Songgot model runs inside the browser (llama.cpp compiled to WebAssembly).
// Nothing leaves the device. The model turns a Korean or English request into one tool call; the app
// then performs the ones it can do locally and hands the rest to the right site when online.
import { Wllama } from "./vendor/wllama/dist/index.js";

const HF_MODEL = "https://huggingface.co/palette-lab/songgot/resolve/main/songgot-nano-q8_0.gguf";
const MODEL_URL = new URLSearchParams(location.search).get("model") === "local" ? new URL("./models/songgot-nano-q8_0.gguf", location.href).href : HF_MODEL;
const MODEL_LABEL = "Songgot-nano Q8_0 · 39M · 42 MB";
const $ = (id) => document.getElementById(id);
const chat = $("chat"), input = $("input"), send = $("send"), status = $("status"), dot = $("dot"), bar = $("bar"), chips = $("chips");
const params = new URLSearchParams(location.search);

let tools = [];
let wllama = null;
let ready = false;

function setStatus(text, mode) { status.textContent = text; dot.className = "dot" + (mode ? " " + mode : ""); }
function setBar(frac) { bar.style.width = (frac == null ? 0 : Math.round(frac * 100)) + "%"; }
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function push(node) { chat.appendChild(node); chat.scrollTop = chat.scrollHeight; return node; }
function say(text, cls = "bot") { return push(el("div", "msg " + cls, text)); }

// ---------- tool routing: the model sees at most 6 candidate tools, chosen by character-bigram overlap ----------
function bigrams(s) { const t = s.replace(/\s+/g, ""); const out = new Set(); for (let i = 0; i < t.length - 1; i++) out.add(t.slice(i, i + 2)); return out; }
function toolText(t) {
  const props = t.parameters && t.parameters.properties ? Object.entries(t.parameters.properties).map(([k, v]) => k + " " + (v.description || "") + " " + (v.enum || []).join(" ")).join(" ") : "";
  return (t.name.replace(/_/g, " ") + " " + (t.description || "") + " " + props).toLowerCase();
}
function pickTools(query, k = 6) {
  const q = bigrams(query.toLowerCase());
  const scored = tools.map((t) => { const b = bigrams(t._text); let s = 0; for (const g of q) if (b.has(g)) s++; return [s, t]; });
  scored.sort((a, b) => b[0] - a[0]);
  const chosen = scored.slice(0, k).map((x) => x[1]);
  return chosen;
}
function stripTool(t) { const { _text, ...rest } = t; return rest; }

// ---------- prompt in the exact training format ----------
function buildPrompt(candidates, query) {
  const toolsJson = JSON.stringify(candidates.map(stripTool));
  return `<|system|>\n${toolsJson}\n<|user|>\n${query.trim()}\n<|call|>\n`;
}

async function callModel(query) {
  const candidates = pickTools(query);
  const prompt = buildPrompt(candidates, query);
  const t0 = performance.now();
  const res = await wllama.createCompletion({ prompt, max_tokens: 160, temperature: 0, stop: ["<|end|>"] });
  const text = (res.choices && res.choices[0] ? res.choices[0].text : String(res)).trim();
  const ms = Math.round(performance.now() - t0);
  let call = null;
  try { call = JSON.parse(text); } catch { const m = text.match(/\{[\s\S]*\}/); if (m) { try { call = JSON.parse(m[0]); } catch {} } }
  return { call, text, ms, candidates, usage: res.usage };
}

// ---------- local actions ----------
const K = (name) => { const t = tools.find((x) => x.name === name); return t ? (t.description || name).replace(/합니다\.?$/, "") : name; };
function fmtArgs(args) { return Object.entries(args || {}).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v, null, 0) : String(v)]); }
function link(label, href, primary) { const a = el("a", primary ? "primary" : "", label); a.href = href; a.target = "_blank"; a.rel = "noopener"; return a; }
function btn(label, fn, primary) { const b = el("button", primary ? "primary" : "", label); b.type = "button"; b.onclick = fn; return b; }
function parseTime(s) { const m = String(s || "").match(/(\d{1,2}):(\d{2})/); return m ? [Number(m[1]), Number(m[2])] : null; }
function scheduleAlarm(args, card) {
  const hm = parseTime(args.time); if (!hm) return say("시간을 HH:MM 형식으로 이해하지 못했어요.", "sys");
  const when = new Date(); when.setHours(hm[0], hm[1], 0, 0); if (when <= new Date()) when.setDate(when.getDate() + 1);
  const label = args.label || "Songgot 알람";
  const fire = () => { try { new Notification(label, { body: `${hm[0]}:${String(hm[1]).padStart(2, "0")} 알람`, icon: "icons/icon-192.png" }); } catch {} ; say(`⏰ ${label} (${args.time})`, "sys"); };
  const go = () => { setTimeout(fire, when - new Date()); say(`알람 예약됨: ${when.toLocaleString("ko-KR")} · 이 화면을 열어 두어야 울립니다.`, "sys"); };
  if ("Notification" in window && Notification.permission !== "granted") Notification.requestPermission().then(go); else go();
}
function startTimer(args) {
  const secs = (Number(args.minutes || 0) * 60) + Number(args.seconds || 0) + (Number(args.hours || 0) * 3600) || Number(args.duration_seconds || 0);
  if (!secs) return say("타이머 길이를 이해하지 못했어요.", "sys");
  const end = Date.now() + secs * 1000; const line = say(`⏳ 타이머 ${secs}초`, "sys");
  const iv = setInterval(() => { const left = Math.max(0, Math.round((end - Date.now()) / 1000)); line.textContent = `⏳ 남은 시간 ${left}초`; if (left <= 0) { clearInterval(iv); line.textContent = "⏰ 타이머 종료"; try { new Notification("타이머 종료", { icon: "icons/icon-192.png" }); } catch {} } }, 500);
}
function icsFor(args) {
  const start = new Date(args.start || args.datetime || args.date || Date.now()); const end = new Date(start.getTime() + 60 * 60 * 1000);
  const f = (d) => d.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
  const body = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Songgot Pocket//KO", "BEGIN:VEVENT", `DTSTART:${f(start)}`, `DTEND:${f(end)}`, `SUMMARY:${args.title || args.summary || "일정"}`, args.location ? `LOCATION:${args.location}` : "", "END:VEVENT", "END:VCALENDAR"].filter(Boolean).join("\r\n");
  return "data:text/calendar;charset=utf-8," + encodeURIComponent(body);
}
function saveNote(args) { const notes = JSON.parse(localStorage.getItem("songgot.notes") || "[]"); notes.push({ t: Date.now(), text: args.content || args.text || args.title || JSON.stringify(args) }); localStorage.setItem("songgot.notes", JSON.stringify(notes)); say(`메모 저장됨 (${notes.length}개)`, "sys"); }
function actionsFor(call) {
  const n = call.name, a = call.arguments || {}, acts = [];
  const q = encodeURIComponent(Object.values(a).filter((v) => typeof v === "string").join(" "));
  if (/alarm/.test(n)) acts.push(btn("알람 예약", () => scheduleAlarm(a), true));
  else if (/timer/.test(n)) acts.push(btn("타이머 시작", () => startTimer(a), true));
  else if (/weather|forecast/.test(n)) acts.push(link("네이버 날씨", `https://search.naver.com/search.naver?query=${encodeURIComponent((a.location || a.city || "") + " 날씨")}`, true));
  else if (/navigat|route|direction|transit|bus|subway|taxi/.test(n)) acts.push(link("네이버 지도", `https://map.naver.com/p/search/${encodeURIComponent(a.destination || a.to || a.query || "")}`, true));
  else if (/music|play|song/.test(n)) acts.push(link("YouTube Music", `https://music.youtube.com/search?q=${q}`, true));
  else if (/email|mail/.test(n)) acts.push(link("메일 열기", `mailto:${a.to || a.recipient || ""}?subject=${encodeURIComponent(a.subject || "")}&body=${encodeURIComponent(a.body || a.content || "")}`, true));
  else if (/message|chat|sms|text/.test(n)) acts.push(link("문자 열기", `sms:${a.to || a.recipient || ""}?&body=${encodeURIComponent(a.message || a.content || a.text || "")}`, true));
  else if (/call|phone|dial/.test(n)) acts.push(link("전화", `tel:${a.number || a.contact || a.to || ""}`, true));
  else if (/calendar|event|schedule|meeting|reminder/.test(n)) acts.push(link("캘린더에 추가 (.ics)", icsFor(a), true));
  else if (/note|memo/.test(n)) acts.push(btn("메모 저장", () => saveNote(a), true));
  else if (/search|lookup|find|news|wiki|translate|exchange|rate|stock|price/.test(n)) acts.push(link("네이버 검색", `https://search.naver.com/search.naver?query=${q}`, true));
  return acts;
}
function renderCall(r) {
  const wrap = el("div", "msg bot");
  if (!r.call || !r.call.name) { wrap.textContent = "응답을 이해하지 못했어요. 다시 말씀해 주세요."; }
  else if (r.call.name === "none") { wrap.textContent = "이 요청에 맞는 도구가 없어요. 알람, 타이머, 날씨, 길찾기, 음악, 메시지, 메일, 일정, 메모 같은 일을 시켜 보세요."; }
  else {
    wrap.textContent = "이렇게 이해했어요:";
    const card = el("div", "card"); const head = el("div", "head", K(r.call.name)); head.appendChild(el("small", "", r.call.name)); card.appendChild(head);
    const body = el("div", "body"); const rows = fmtArgs(r.call.arguments); if (!rows.length) body.appendChild(el("span", "", "인자 없음")); for (const [k, v] of rows) { body.appendChild(el("b", "", k)); body.appendChild(el("span", "", v)); } card.appendChild(body);
    const acts = actionsFor(r.call); if (acts.length) { const act = el("div", "act"); acts.forEach((x) => act.appendChild(x)); card.appendChild(act); }
    wrap.appendChild(card);
  }
  const det = el("details", "raw"); const u = r.usage || {}; const gen = u.completion_tokens || 0; const tps = gen && r.ms ? (gen / (r.ms / 1000)).toFixed(1) : "-";
  wrap.appendChild(el("div", "meta", `${u.prompt_tokens || "-"} prompt tok · ${gen} gen tok · ${tps} tok/s · ${r.ms} ms · on device`));
  det.appendChild(el("summary", "", `모델 출력 · 후보 도구 ${r.candidates.map((t) => t.name).join(", ")}`)); const pre = el("pre", "", r.text); det.appendChild(pre); wrap.appendChild(det);
  push(wrap);
  if (r.call && r.call.name && r.call.name !== "none") followups(r.call.name);
  return wrap;
}
function followups(name) {
  const F = [[/alarm|timer|reminder/, ["30분 뒤로 미뤄줘", "알람 취소해줘", "매일 반복해줘"]], [/weather/, ["내일은?", "주말 날씨 알려줘", "미세먼지는?"]],
             [/navigat|transit|route|bus|subway|taxi/, ["차로 가면 얼마나 걸려?", "택시 불러줘", "근처 맛집 찾아줘"]], [/music|play|song/, ["다음 곡", "볼륨 줄여줘", "잔잔한 노래 틀어줘"]],
             [/message|chat|sms|mail|call|phone/, ["10분 늦는다고 보내줘", "아빠한테도 보내줘", "전화 걸어줘"]], [/calendar|event|schedule|meeting/, ["다음 주 같은 시간에도 잡아줘", "장소는 강남으로", "내일 일정 알려줘"]]];
  const hit = F.find(([re]) => re.test(name)); if (!hit) return;
  const row = el("div", "chips"); row.style.padding = "0"; row.style.alignSelf = "flex-start";
  hit[1].forEach((t) => { const c = el("button", "chip", t); c.type = "button"; c.onclick = () => { row.remove(); handle(t); }; row.appendChild(c); });
  push(row);
}

// ---------- chat loop ----------
async function handle(query) {
  if (!ready || !query.trim()) return;
  say(query, "user"); input.value = ""; send.disabled = true; setStatus("thinking", "busy");
  try { const r = await callModel(query); renderCall(r); if (params.get("q")) document.title = "RESULT " + r.text; }
  catch (e) { say("오류: " + (e && e.message ? e.message : e), "sys"); if (params.get("q")) document.title = "ERROR " + e; }
  setStatus(navigator.onLine ? "ready · on device" : "ready · offline", "ok"); send.disabled = false; input.focus();
}
$("form").addEventListener("submit", (e) => { e.preventDefault(); handle(input.value); });
["내일 아침 7시에 알람 맞춰줘", "부산 날씨 어때?", "강남역까지 대중교통으로 안내해 줘", "10분 타이머", "엄마한테 늦는다고 문자 보내줘", "빗소리 틀어줘"].forEach((s) => { const c = el("button", "chip", s); c.type = "button"; c.onclick = () => handle(s); chips.appendChild(c); });
window.addEventListener("online", () => ready && setStatus("ready · on device", "ok"));
window.addEventListener("offline", () => ready && setStatus("ready · offline", "ok"));

// ---------- boot ----------
(async () => {
  say("Songgot Pocket. 말한 것을 실행할 행동으로 바꿔 주는 기기 내장 비서입니다. 첫 실행 때 모델(42 MB)을 한 번 받아 두면 그 뒤로는 인터넷 없이 동작합니다.", "sys");
  try {
    tools = (await (await fetch("./tools.json")).json()).map((t) => ({ ...t, _text: toolText(t) }));
    setStatus("loading model", "busy");
    wllama = new Wllama({ default: new URL("./vendor/wllama/dist/wllama.wasm", location.href).href }, { allowOffline: true, suppressNativeLog: true });
    await wllama.loadModelFromUrl(MODEL_URL, { n_ctx: 1024, n_batch: 512, progressCallback: ({ loaded, total }) => { setBar(total ? loaded / total : null); setStatus(`downloading ${Math.round(100 * loaded / (total || 1))}%`, "busy"); } });
    setBar(1); ready = true; send.disabled = false;
    setStatus(navigator.onLine ? "ready · on device" : "ready · offline", "ok");
    say(`모델 준비 완료 (${MODEL_LABEL}, ${wllama.isMultithread() ? "multi-thread" : "single-thread"}). 무엇을 할까요?`, "sys");
    if ("serviceWorker" in navigator) navigator.serviceWorker.register("./sw.js").catch(() => {});
    if (params.get("q")) handle(params.get("q"));
  } catch (e) { setStatus("failed", ""); const msg = String(e && e.message ? e.message : e); say(/404|not found/i.test(msg) ? "가중치가 아직 올라오지 않았어요 (훈련 완료 후 오늘 업로드됩니다). 잠시 후 다시 열어 주세요." : "모델을 불러오지 못했어요: " + msg, "sys"); if (params.get("q")) document.title = "ERROR " + e; }
})();
