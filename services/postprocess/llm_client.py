import os
import requests
from typing import Optional


def process_with_groq(raw_tokens: str) -> str:
    """
    Process raw ISL tokens using Groq API to generate coherent sentences.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key or groq_api_key == "<APIKEY>":
        raise ValueError("GROQ_API_KEY not set in environment variables")

    # Create a comprehensive prompt for ISL text normalization
    prompt = f"""You are an expert in Indian Sign Language (ISL) text processing. Your task is to convert raw ISL tokens into coherent, grammatically correct English sentences.

ISL tokens are typically individual words or phrases without connecting words, articles, or proper sentence structure. Your job is to:

1. Identify the core meaning and intent
2. Add appropriate connecting words, articles, and grammar
3. Create natural, fluent English sentences
4. Maintain the original meaning while making it readable

Raw ISL tokens: "{raw_tokens}"

Please convert this into a coherent English sentence. If the input is already well-formed, just return it as-is. If it's unclear or incomplete, make your best interpretation.

Response (only the processed text, no explanations):"""

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 200,
        "stream": False
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(
                f"Groq API error: {response.status_code} - {response.text}")

        result = response.json()

        if "choices" not in result or not result["choices"]:
            raise Exception("No response from Groq API")

        processed_text = result["choices"][0]["message"]["content"].strip()

        # Fallback if empty response
        if not processed_text:
            processed_text = raw_tokens

        return processed_text

    except requests.exceptions.Timeout:
        raise Exception("Request to Groq API timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing with Groq: {str(e)}")
