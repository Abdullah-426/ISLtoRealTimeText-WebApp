from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from llm_client import process_with_groq

# Load environment variables
load_dotenv()

app = FastAPI(title="ISL Post-processing Service", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PostProcessRequest(BaseModel):
    raw_tokens: str
    lang: str = "en"


class PostProcessResponse(BaseModel):
    text: str
    success: bool
    error: str = ""


@app.get("/")
def health_check():
    return {"status": "healthy", "service": "postprocess"}


@app.post("/postprocess", response_model=PostProcessResponse)
def postprocess_text(request: PostProcessRequest):
    try:
        if not request.raw_tokens.strip():
            return PostProcessResponse(
                text="",
                success=True,
                error=""
            )

        # Process with Groq API
        processed_text = process_with_groq(request.raw_tokens)

        return PostProcessResponse(
            text=processed_text,
            success=True,
            error=""
        )

    except Exception as e:
        return PostProcessResponse(
            text="",
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
