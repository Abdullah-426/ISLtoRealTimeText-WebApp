from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import os
from fastapi.middleware.cors import CORSMiddleware
from llm_client import normalize_with_llm, RuleBasedNormalizer

app = FastAPI(title='ISL Postprocess', version='0.2.0')
app.add_middleware(CORSMiddleware, allow_origins=[
                   '*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

SYSTEM_PROMPT = (
    "You are a text normalizer. Input: tokenized core words from Indian Sign Language. "
    "Output: one natural sentence with minimal extra words, correct punctuation, and unchanged facts. "
    "Do not hallucinate new information."
)


class PostReq(BaseModel):
    raw_tokens: Union[List[str],
                      str] = Field(..., description='Committed tokens or raw string')
    lang: Optional[str] = Field(default='en', description='en|hi (optional)')
    style: Optional[str] = Field(
        default='simple', description='simple|formal (optional)')


class PostResp(BaseModel):
    text: str
    note: Optional[str] = None


@app.get('/')
def health():
    return {"ok": True}


@app.post('/postprocess', response_model=PostResp)
async def postprocess(req: PostReq):
    provider = os.getenv('LLM_PROVIDER', 'local').lower()
    openai_key = os.getenv('OPENAI_API_KEY')
    groq_key = os.getenv('GROQ_API_KEY')
    local_url = os.getenv('LOCAL_GPT_URL', 'http://localhost:8001')

    raw_str = req.raw_tokens if isinstance(
        req.raw_tokens, str) else ' '.join(req.raw_tokens)
    raw_str = raw_str.strip()
    if not raw_str:
        raise HTTPException(status_code=400, detail='raw_tokens required')

    if provider == 'openai' and not openai_key:
        text = RuleBasedNormalizer()(raw_str)
        return PostResp(text=text, note='OPENAI_API_KEY missing; used rule-based fallback')
    if provider == 'groq' and not groq_key:
        text = RuleBasedNormalizer()(raw_str)
        return PostResp(text=text, note='GROQ_API_KEY missing; used rule-based fallback')

    text = await normalize_with_llm(
        raw=raw_str, provider=provider, openai_key=openai_key, groq_key=groq_key, local_url=local_url,
        system_prompt=SYSTEM_PROMPT, lang=req.lang or 'en', style=req.style or 'simple'
    )
    return PostResp(text=text)
