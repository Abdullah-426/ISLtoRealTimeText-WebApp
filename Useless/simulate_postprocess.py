import argparse
import json
import sys
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:8000/postprocess')
parser.add_argument('--tokens', required=False,
                    default='["HELLO","HOW","YOU"]', help='JSON array of tokens')
parser.add_argument('--lang', default='en')
args = parser.parse_args()

try:
    raw_tokens = json.loads(args.tokens)
except Exception as e:
    print('Invalid --tokens JSON:', e)
    sys.exit(1)

payload = {"raw_tokens": raw_tokens, "lang": args.lang}
resp = requests.post(args.url, json=payload, timeout=20)
print(resp.status_code)
print(resp.text)
