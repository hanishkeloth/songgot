"""KoSGD (AIWORKX, CC BY-SA 4.0: human-reviewed Korean translation of Google's Schema-Guided Dialogue) -> Songgot
tool-call rows. Each service intent becomes a tool whose parameters are the schema's slots; each FIRST user turn
of a dialogue becomes one row: query = the Korean utterance, call = {active intent, slot values stated in that
turn}. Only first turns are used so the row is single-turn like the benchmark. Human-reviewed slot values are the
argument-extraction supervision our failure analysis asked for (2026-09-12: 203 of 500 misses were right tool,
wrong argument value).

    modal deploy harness/kosgd_to_tools.py
    .venv/bin/python harness/spawn.py songgot-kosgd convert
"""
import json
import os
import re
import time

import modal

app = modal.App("songgot-kosgd")
vol = modal.Volume.from_name("songgot", create_if_missing=True)
V = "/vol"
image = modal.Image.debian_slim(python_version="3.11")
ROOT = f"{V}/corpora/hf/AIWORKX__KoSGD"


def find(pattern):
    out = []
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if re.search(pattern, f):
                out.append(os.path.join(d, f))
    return sorted(out)


def schema_tools(schema_list):
    """One tool per intent: name = intent name, parameters = the intent's required + optional slots."""
    tools, slot_desc = {}, {}
    for svc in schema_list:
        sname = svc.get("service_name", "")
        slots = {s["name"]: s for s in svc.get("slots", [])}
        for it in svc.get("intents", []):
            props = {}
            for sl in list(it.get("required_slots", [])) + list((it.get("optional_slots") or {}).keys()):
                s = slots.get(sl, {})
                props[sl] = {"type": "string", "description": s.get("description", "")}
                if s.get("possible_values") and s.get("is_categorical"):
                    props[sl]["enum"] = s["possible_values"]
            tools[(sname, it["name"])] = {"name": it["name"], "description": it.get("description", ""),
                                          "parameters": {"type": "object", "properties": props, "required": list(it.get("required_slots", []))}}
    return tools


@app.function(image=image, volumes={V: vol}, cpu=4, memory=16384, timeout=60 * 60)
def convert(out: str = "sft/kosgd_calls.jsonl", max_tools: int = 8):
    vol.reload(); t0 = time.time()
    schema_files = find(r"schema.*\.json$"); dlg_files = [f for f in find(r"dialogues.*\.json$") if "schema" not in f]
    print(f"[kosgd] {len(schema_files)} schema files, {len(dlg_files)} dialogue files", flush=True)
    tools_by_split = {}
    for sf in schema_files:
        tools_by_split[os.path.dirname(sf)] = schema_tools(json.load(open(sf, encoding="utf-8")))
    n = skipped = 0; services = set()
    with open(f"{V}/{out}", "w", encoding="utf-8") as fo:
        for df in dlg_files:
            tools = tools_by_split.get(os.path.dirname(df)) or next(iter(tools_by_split.values()))
            for dlg in json.load(open(df, encoding="utf-8")):
                turns = dlg.get("turns", [])
                user = next((t for t in turns if t.get("speaker") == "USER"), None)
                if not user:
                    skipped += 1; continue
                q = user.get("utterance", "").strip()
                frame = next((f for f in user.get("frames", []) if f.get("state", {}).get("active_intent") not in (None, "", "NONE")), None)
                if not frame or not q or not re.search(r"[가-힣]", q):
                    skipped += 1; continue
                svc, intent = frame["service"], frame["state"]["active_intent"]
                if (svc, intent) not in tools:
                    skipped += 1; continue
                args = {k: (v[0] if isinstance(v, list) and v else v) for k, v in (frame["state"].get("slot_values") or {}).items()}
                args = {k: v for k, v in args.items() if isinstance(v, str) and v and v != "dontcare"}
                # tools shown: every intent of the dialogue's services, capped
                shown = [tools[k] for k in tools if k[0] in set(dlg.get("services", [svc]))][:max_tools]
                if not any(t["name"] == intent for t in shown):
                    shown = [tools[(svc, intent)]] + shown[: max_tools - 1]
                fo.write(json.dumps({"lang": "ko", "query": q, "tools": shown, "call": {"name": intent, "arguments": args}, "cond": "kosgd", "src": "AIWORKX/KoSGD CC BY-SA 4.0"}, ensure_ascii=False) + "\n")
                n += 1; services.add(svc)
    vol.commit()
    msg = f"[kosgd] DONE {n} rows ({skipped} skipped) over {len(services)} services in {time.time()-t0:.0f}s -> {out}"
    print(msg, flush=True); open(f"{V}/corpora/PROVENANCE.md", "a", encoding="utf-8").write(time.strftime("%F %T ") + msg + "\n"); vol.commit()
    return {"rows": n, "skipped": skipped}
