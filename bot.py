import os
import json
import time
import uuid
import datetime
import traceback
import re
import io
import urllib.request
import urllib.parse
import contextlib
from threading import Thread

import requests
from flask import Flask, send_file, jsonify
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
LLM_API_KEY = (
    os.environ.get("GROQ_API_KEY")
    or os.environ.get("GROK_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)
if not LLM_API_KEY:
    raise SystemExit("Set GROQ_API_KEY (recommended) or GROK_API_KEY or OPENAI_API_KEY")

LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://localhost:10000")
LOG_FILE = os.environ.get("LOG_FILE", "run.jsonl")
LOG_URL = f"{PUBLIC_BASE_URL.rstrip('/')}/run.jsonl"
PORT = int(os.environ.get("PORT", 10000))

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
TG_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# ─── Logging ─────────────────────────────────────────────────────
def log_run(entry: dict):
    entry["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print("log write failed:", e)


# ─── Tools ───────────────────────────────────────────────────────
def fetch_url(url: str, max_bytes: int = 2_000_000) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; tds-p1-bot/1.0)",
                "Accept": "*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read(max_bytes)
        return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"[fetch_url ERROR] {e}"


def run_python(code: str) -> str:
    buf = io.StringIO()
    safe_globals = {
        "__builtins__": __builtins__,
        "json": json,
        "re": re,
        "math": __import__("math"),
        "statistics": __import__("statistics"),
        "urllib": urllib,
        "requests": requests,
        "datetime": datetime,
    }
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, safe_globals)
        out = buf.getvalue()
        return out[:5000] if out else "[no stdout]"
    except Exception as e:
        return f"[python ERROR] {e}\n{traceback.format_exc()[:2000]}"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Download the contents of a URL (HTML/JSON/CSV/text).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return stdout. Modules: json, re, math, statistics, urllib, requests, datetime.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a data-analyst agent. You MUST reply with EXACTLY ONE JSON OBJECT
and nothing else — no prose, no greetings, no explanations, no markdown fences.

The user's message contains a question. It may or may not specify an exact JSON shape.

Rules for your response:
1. If the message specifies a JSON shape (e.g. 'Reply with ONLY this JSON: {"state": "<name>"}'),
   respond with EXACTLY that shape filled in with the correct answer.

2. If the message asks a clear question but does NOT specify a shape, choose a sensible key
   name yourself and respond with a single-key JSON object.
   Examples:
     "What is the capital of France?"  →  {"capital": "Paris"}
     "What is the largest planet?"      →  {"planet": "Jupiter"}
     "How many states are in India?"    →  {"count": 28}
     "Who wrote Hamlet?"                →  {"author": "William Shakespeare"}

3. If the message is a pure greeting with NO question ("hi", "hello", "test"),
   reply with: {"error": "no question provided"}

4. If you genuinely cannot determine the answer (future events, ambiguous, no data),
   reply with: {"error": "<short reason>"}

5. NEVER output prose. NEVER greet. NEVER explain.
6. NEVER wrap JSON in ```json fences.
7. NEVER include "answer" or "log_url" keys — the harness wraps your JSON.

Workflow when you need external data:
- Use fetch_url to load public data (Wikipedia, MOSPI, data.gov.in, World Bank, RBI, WHO, etc.)
- Use run_python to parse HTML/JSON or compute aggregations.
- Keep tool calls ≤ 6.
- Prefer official primary sources for statistical questions."""


def agent_answer(question: str, run_id: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(8):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as e:
            log_run({"run_id": run_id, "phase": f"llm_error_{step}", "error": str(e)})
            return json.dumps({"error": f"llm api error: {e}"})

        msg = resp.choices[0].message
        log_run({
            "run_id": run_id,
            "phase": f"llm_step_{step}",
            "content": (msg.content or "")[:1500],
            "tool_calls": [
                {"name": tc.function.name, "args": tc.function.arguments[:500]}
                for tc in (msg.tool_calls or [])
            ],
        })

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                if name == "fetch_url":
                    result = fetch_url(args.get("url", ""))
                elif name == "run_python":
                    result = run_python(args.get("code", ""))
                else:
                    result = f"[unknown tool: {name}]"

                log_run({
                    "run_id": run_id,
                    "phase": f"tool_result_{step}",
                    "tool": name,
                    "args": args,
                    "output_preview": result[:1500],
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:8000],
                })
            continue

        final = (msg.content or "").strip()
        final = re.sub(r"^```(?:json)?\s*", "", final)
        final = re.sub(r"\s*```$", "", final).strip()
        log_run({"run_id": run_id, "phase": "final", "answer": final})
        return final

    log_run({"run_id": run_id, "phase": "step_limit_hit"})
    return json.dumps({"error": "agent_step_limit"})


def build_reply(final_answer_str: str) -> str:
    try:
        parsed = json.loads(final_answer_str)
        if isinstance(parsed, dict) and "answer" in parsed and "log_url" in parsed:
            return json.dumps(parsed, ensure_ascii=False)
        answer_value = parsed
    except Exception:
        answer_value = final_answer_str
    return json.dumps({"answer": answer_value, "log_url": LOG_URL}, ensure_ascii=False)


# ─── Telegram ────────────────────────────────────────────────────
def tg_get_updates(offset=None, timeout=30):
    params = {"timeout": timeout}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"{TG_API}/getUpdates", params=params, timeout=timeout + 10)
    return r.json()


def tg_send(chat_id: int, text: str):
    if len(text) > 4000:
        text = text[:4000]
    requests.post(
        f"{TG_API}/sendMessage",
        json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        timeout=20,
    )

def handle_message(msg: dict):
    text = (msg.get("text") or msg.get("caption") or "").strip()
    chat_id = msg["chat"]["id"]
    user = msg.get("from", {}).get("username", "?")
    run_id = uuid.uuid4().hex[:12]

    log_run({"run_id": run_id, "phase": "incoming", "chat_id": chat_id, "user": user, "text": text})

    if not text:
        return

    # Commands
    if text in ("/start", "/help"):
        tg_send(chat_id, json.dumps({
            "answer": "Send a data-analysis question that specifies the JSON shape you want back.",
            "log_url": LOG_URL,
        }))
        return

    # Only skip if it's clearly nothing but a greeting
    if text.lower().strip(".!? ") in ("hi", "hello", "hey"):
        tg_send(chat_id, json.dumps({
            "answer": {"error": "no question provided"},
            "log_url": LOG_URL,
        }))
        return

    try:
        final = agent_answer(text, run_id)
    except Exception as e:
        log_run({"run_id": run_id, "phase": "agent_error", "error": str(e), "trace": traceback.format_exc()[:2000]})
        final = json.dumps({"error": str(e)[:200]})

    reply = build_reply(final)
    log_run({"run_id": run_id, "phase": "outgoing", "reply": reply})
    tg_send(chat_id, reply)
# ─── HTTP server ─────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def health():
    return "ok"


@app.route("/run.jsonl")
def serve_log():
    try:
        return send_file(LOG_FILE, mimetype="application/x-ndjson")
    except Exception:
        return "", 200


@app.route("/status")
def status():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
    except Exception:
        lines = 0
    return jsonify({"ok": True, "log_lines": lines, "log_url": LOG_URL, "model": LLM_MODEL})


def run_http():
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


# ─── Main ────────────────────────────────────────────────────────
def main():
    print(f"Bot starting. LOG_URL={LOG_URL}  MODEL={LLM_MODEL}")
    log_run({"phase": "boot", "log_url": LOG_URL, "model": LLM_MODEL})
    Thread(target=run_http, daemon=True).start()
    time.sleep(1)

    last_update = None
    while True:
        try:
            updates = tg_get_updates(offset=last_update)
            if not updates.get("ok"):
                time.sleep(2)
                continue
            for upd in updates.get("result", []):
                last_update = upd["update_id"] + 1
                if "message" in upd:
                    try:
                        handle_message(upd["message"])
                    except Exception as e:
                        print("handler error:", e)
                        traceback.print_exc()
        except Exception as e:
            print("loop error:", e)
            time.sleep(3)


if __name__ == "__main__":
    main()