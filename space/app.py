"""Songgot demo: Korean tool calling on CPU through llama.cpp, the same runtime and tokenizer as the
Pocket app and every GGUF user. Paste tool schemas (JSON list) and a request."""
import json, os
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

REPO = os.environ.get("SONGGOT_REPO", "palette-lab/songgot")  # tokenizer (vocab-only GGUF)
WEIGHTS_REPO = os.environ.get("SONGGOT_WEIGHTS_REPO", "palette-lab/songgot-12l")  # the 12-layer model, 11.4 percent on FunctionChat SingleCall


def _get(name, repo=None):
    try:
        return hf_hub_download(repo or REPO, name)
    except Exception as e:  # file not uploaded yet
        print("missing", name, e)
        return None


vocab = _get("songgot-vocab.gguf")
weights = _get("songgot-q8_0.gguf", WEIGHTS_REPO)
tok = Llama(model_path=vocab, vocab_only=True, verbose=False) if vocab else None
llm = Llama(model_path=weights, n_ctx=1024, n_threads=2, verbose=False) if weights else None

DEFAULT_TOOLS = json.dumps([
    {"name": "set_alarm", "description": "알람을 설정합니다.", "parameters": {"type": "object", "properties": {"time": {"type": "string", "description": "HH:MM"}, "label": {"type": "string"}}, "required": ["time"]}},
    {"name": "get_weather", "description": "지역의 날씨를 조회합니다.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "date": {"type": "string"}}, "required": ["location"]}},
    {"name": "navigate_to", "description": "목적지까지 길 안내를 시작합니다.", "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "mode": {"type": "string", "enum": ["car", "transit", "walk"]}}, "required": ["destination"]}},
    {"name": "order_food", "description": "배달 음식을 주문합니다.", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}, "quantity": {"type": "integer"}}, "required": ["items"]}},
], ensure_ascii=False, indent=1)


def call(tools_json: str, query: str):
    if llm is None:
        return "가중치 업로드 중입니다 (weights are uploaded when training finishes). 토크나이저 탭은 지금 동작합니다."
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError as e:
        return f"tools JSON error: {e}"
    prompt = "<|system|>\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":")) + f"\n<|user|>\n{query.strip()}\n<|call|>\n"
    out = llm(prompt, max_tokens=160, temperature=0.0, stop=["<|end|>"])
    text = out["choices"][0]["text"].strip()
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=1)
    except json.JSONDecodeError:
        return text


HANGUL = lambda ch: 0xAC00 <= ord(ch) <= 0xD7A3


def tokenize(query: str):
    if tok is None:
        return "tokenizer file not uploaded yet"
    ids = tok.tokenize(query.encode("utf-8"), add_bos=False, special=True)
    pieces = [tok.detokenize([i], special=True).decode("utf-8", "replace") for i in ids]
    syl = sum(1 for ch in query if HANGUL(ch))
    ratio = f"{len(ids)/syl:.2f} tokens per Hangul syllable" if syl else "no Hangul in the query"
    return f"{len(ids)} tokens for {len(query)} characters ({ratio})\n\n" + " | ".join(p.replace(" ", "▁") for p in pieces)


call_demo = gr.Interface(
    fn=call,
    inputs=[gr.Textbox(value=DEFAULT_TOOLS, lines=14, label="도구 (JSON 목록)"), gr.Textbox(value="내일 아침 7시에 알람 맞춰줘", label="요청")],
    outputs=gr.Textbox(lines=6, label="호출"),
    title="Songgot (송곳): Korean-first tiny agentic model",
    description="A from-scratch tiny model for Korean tool calling on the device, run here through llama.cpp on CPU. Apache 2.0. Paper and code: github.com/hanishkeloth/songgot. 이 데모는 CPU에서 실행됩니다.",
    examples=[[DEFAULT_TOOLS, "부산 날씨 어때?"], [DEFAULT_TOOLS, "강남역까지 대중교통으로 안내해 주세요"], [DEFAULT_TOOLS, "치킨 두 마리 시켜줘"], [DEFAULT_TOOLS, "오늘 하루 어땠어?"]],
    allow_flagging="never",
)
tok_demo = gr.Interface(
    fn=tokenize,
    inputs=gr.Textbox(value="내일 아침 7시에 알람 맞춰줘", label="문장"),
    outputs=gr.Textbox(lines=4, label="Songgot 32k tokenizer (llama.cpp)"),
    title="Songgot tokenizer: 0.90 tokens per Hangul syllable",
    description="Measured on 100 FunctionChat SingleCall queries: Songgot 0.90, Gemma 3 0.98, Qwen3 1.15, Needle 2 3.47 tokens per syllable.",
    examples=[["부산 날씨 어때?"], ["강남역까지 대중교통으로 안내해 주세요"], ["Set an alarm for 7 tomorrow morning"]],
    allow_flagging="never",
)
demo = gr.TabbedInterface([call_demo, tok_demo], ["도구 호출 (tool call)", "토크나이저 (tokenizer)"], title="Songgot (송곳)")
demo.launch()
