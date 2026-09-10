"""Songgot demo: Korean tool calling on CPU. Paste tool schemas (JSON list) and a request."""
import json, os
import gradio as gr
import torch, sentencepiece as spm
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

REPO = os.environ.get("SONGGOT_REPO", "palette-lab/songgot")
d = snapshot_download(REPO)
sp = spm.SentencePieceProcessor(model_file=os.path.join(d, "tokenizer.model"))
END, PAD, EOS, BOS = sp.encode("<|end|>")[-1], sp.encode("<|pad|>")[-1], sp.eos_id(), sp.bos_id()
try:
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()
except Exception as e:  # weights not uploaded yet: tokenizer demo still works
    print("model load failed:", e)
    model = None

HANGUL = lambda ch: 0xAC00 <= ord(ch) <= 0xD7A3

def tokenize(query: str):
    pieces = sp.encode(query, out_type=str)
    syl = sum(1 for ch in query if HANGUL(ch))
    ratio = f"{len(pieces)/syl:.2f} tokens per Hangul syllable" if syl else "no Hangul in the query"
    return f"{len(pieces)} tokens for {len(query)} characters ({ratio})\n\n" + " | ".join(pieces)

DEFAULT_TOOLS = json.dumps([
    {"name": "set_alarm", "description": "알람을 설정합니다.", "parameters": {"type": "object", "properties": {"time": {"type": "string", "description": "HH:MM"}, "label": {"type": "string"}}, "required": ["time"]}},
    {"name": "get_weather", "description": "지역의 날씨를 조회합니다.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "date": {"type": "string"}}, "required": ["location"]}},
    {"name": "navigate_to", "description": "목적지까지 길 안내를 시작합니다.", "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "mode": {"type": "string", "enum": ["car", "transit", "walk"]}}, "required": ["destination"]}},
    {"name": "order_food", "description": "배달 음식을 주문합니다.", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}, "quantity": {"type": "integer"}}, "required": ["items"]}},
], ensure_ascii=False, indent=1)


def call(tools_json: str, query: str):
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError as e:
        return f"tools JSON error: {e}"
    if model is None:
        return "가중치 업로드 중입니다 (weights are being uploaded after training on 2026-09-10). 토크나이저 탭은 지금 동작합니다."
    prompt = "<|system|>\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")) + f"\n<|user|>\n{query.strip()}\n<|call|>\n"
    ids = torch.tensor([[BOS] + sp.encode(prompt)])
    with torch.no_grad():
        g = model.generate(input_ids=ids, max_new_tokens=160, do_sample=False, eos_token_id=[END, EOS], pad_token_id=PAD)
    out = sp.decode([int(x) for x in g[0][ids.shape[1]:].tolist() if int(x) not in (END, EOS, PAD)]).strip()
    try:
        return json.dumps(json.loads(out), ensure_ascii=False, indent=1)
    except json.JSONDecodeError:
        return out

call_demo = gr.Interface(
    fn=call,
    inputs=[gr.Textbox(value=DEFAULT_TOOLS, lines=14, label="도구 (JSON 목록)"), gr.Textbox(value="내일 아침 7시에 알람 맞춰줘", label="요청")],
    outputs=gr.Textbox(lines=6, label="호출"),
    title="Songgot (송곳): Korean-first tiny agentic model",
    description="A from-scratch tiny model for Korean tool calling on the device. Apache 2.0. Paper and code: github.com/hanishkeloth/songgot. 이 데모는 CPU에서 실행됩니다.",
    examples=[[DEFAULT_TOOLS, "부산 날씨 어때?"], [DEFAULT_TOOLS, "강남역까지 대중교통으로 안내해 주세요"], [DEFAULT_TOOLS, "치킨 두 마리 시켜줘"], [DEFAULT_TOOLS, "오늘 하루 어땠어?"]],
    allow_flagging="never",
)
tok_demo = gr.Interface(
    fn=tokenize,
    inputs=gr.Textbox(value="내일 아침 7시에 알람 맞춰줘", label="문장"),
    outputs=gr.Textbox(lines=4, label="Songgot 32k tokenizer"),
    title="Songgot tokenizer: 0.90 tokens per Hangul syllable",
    description="Measured on 100 FunctionChat SingleCall queries: Songgot 0.90, Gemma 3 0.98, Qwen3 1.15, Needle 2 3.47 tokens per syllable.",
    examples=[["부산 날씨 어때?"], ["강남역까지 대중교통으로 안내해 주세요"], ["Set an alarm for 7 tomorrow morning"]],
    allow_flagging="never",
)
demo = gr.TabbedInterface([call_demo, tok_demo], ["도구 호출 (tool call)", "토크나이저 (tokenizer)"], title="Songgot (송곳)")
demo.launch()
