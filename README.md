# TDS P1 — Data Analyst Telegram Bot (Grok)

Answers data-analysis questions with exactly one JSON object, using Grok (xAI) as the LLM.

## Environment variables

| Variable | Description |
|---|---|
| `TELEGRAM_TOKEN` | From [@BotFather](https://t.me/BotFather) |
| `GROK_API_KEY` | From https://console.x.ai/team/default/api-keys |
| `PUBLIC_BASE_URL` | Public base URL of the deployment (e.g. `https://myapp.onrender.com`) |
| `GROK_MODEL` | (optional) default `grok-2-1212` |
| `PORT` | (optional) default `10000` |

## Run locally

```bash
pip install -r requirements.txt
export TELEGRAM_TOKEN=...
export GROK_API_KEY=...
export PUBLIC_BASE_URL=http://localhost:10000
python bot.py