import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

def generate_completion(messages: list, task_type: str = "fast", max_tokens: int = 150, temperature: float = 0.0) -> str:
    """
    Generate an LLM completion using the configured provider.
    
    Args:
        messages: List of message dicts (e.g., [{"role": "user", "content": "..."}])
        task_type: "fast" (for classification/escalation) or "draft" (for generating replies)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        # Local or Remote Ollama Execution
        model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        base_url_raw = os.getenv("OLLAMA_BASE_URL", "localhost:11434")
        
        # Ensure proper URL formatting for the OpenAI client
        if not base_url_raw.startswith("http"):
            base_url_raw = f"http://{base_url_raw}"
        if not base_url_raw.endswith("/v1"):
            base_url_raw = f"{base_url_raw}/v1"

        client = OpenAI(
            base_url=base_url_raw,
            api_key="ollama", # required, but unused by ollama
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body={"think": False}
        )
        return response.choices[0].message.content.strip()

    else:
        # Groq Execution
        client = Groq()
        # Fallbacks to default groq models
        if task_type == "draft":
            model = "qwen/qwen3.8-27b" # originally gpt-oss-120b but changed to prevent TPM limits
        else:
            model = "qwen/qwen3.8-27b"

        # Rate Limit / Retry logic could go here if needed, 
        # but evaluation_harness/automated_metrics.py sleeps mitigate this on the batch level.
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
