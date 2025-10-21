from typing import Optional
import os
import httpx


class RuleBasedNormalizer:
    def __call__(self, raw: str) -> str:
        s = raw.strip().upper()
        if s in {'HELLO HOW YOU', 'HELLO HOW ARE YOU'}:
            return 'Hello, how are you?'
        if s == 'WHAT NAME YOU':
            return 'What is your name?'
        if s == 'WHERE TOILET':
            return 'Where is the toilet?'
        out = raw.strip().capitalize()
        if not out.endswith(('.', '!', '?')):
            out += '.'
        return out


async def normalize_with_llm(*, raw: str, provider: str, openai_key: Optional[str], groq_key: Optional[str], local_url: str, system_prompt: str, lang: str, style: str) -> str:
    provider = provider.lower()
    if provider == 'local':
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(local_url, json={"system": system_prompt, "input": raw, "lang": lang, "style": style})
                if r.status_code == 200:
                    return (r.json().get('text') or r.text).strip()
        except Exception:
            pass
        return RuleBasedNormalizer()(raw)

    if provider == 'openai':
        assert openai_key, 'OPENAI_API_KEY required'
        headers = {"Authorization": f"Bearer {openai_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"lang={lang}; style={style}; raw={raw}"}
            ],
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data['choices'][0]['message']['content'].strip()

    if provider == 'groq':
        assert groq_key, 'GROQ_API_KEY required'
        headers = {"Authorization": f"Bearer {groq_key}",
                   "Content-Type": "application/json"}
        payload = {
            "model": os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile'),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"lang={lang}; style={style}; raw={raw}"}
            ],
            "temperature": 0.2
        }
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data['choices'][0]['message']['content'].strip()

    return RuleBasedNormalizer()(raw)
