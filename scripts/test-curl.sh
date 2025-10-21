#!/usr/bin/env bash
set -e
URL=${1:-http://localhost:8000/postprocess}

echo "POST to $URL"
curl -s -X POST "$URL" \
  -H 'Content-Type: application/json' \
  -d '{"raw_tokens":["HELLO","HOW","YOU"],"lang":"en"}' | jq .
