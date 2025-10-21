#!/usr/bin/env bash
set -euo pipefail

# Start phrase-infer backend
(
  cd services/infer
  if [ ! -d .venv ]; then
    python -m venv .venv
  fi
  source .venv/bin/activate || source .venv/Scripts/activate
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8001 &
)

# Start postprocess backend
(
  cd services/postprocess
  if [ ! -d .venv ]; then
    python -m venv .venv
  fi
  source .venv/bin/activate || source .venv/Scripts/activate
  pip install -r requirements.txt
  export LLM_PROVIDER=${LLM_PROVIDER:-local}
  uvicorn main:app --host 0.0.0.0 --port 8000 &
)

# Start frontend
(
  cd frontend
  if command -v pnpm >/dev/null 2>&1; then
    pnpm i && pnpm dev
  elif command -v yarn >/dev/null 2>&1; then
    yarn && yarn dev
  else
    npm i && npm run dev
  fi
)
