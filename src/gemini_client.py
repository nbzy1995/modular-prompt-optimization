from typing import Optional
from google import genai
import os, time

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = model_name
        self.last_request_time = 0.0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0,
        stream: bool = False,
        rate_limit: bool = True
    ) -> str:
        """Send prompt to Gemini and return text output."""

        # --- rate limiting ---
        if rate_limit:
            elapsed = time.time() - self.last_request_time
            if elapsed < 4:  # ~15 requests/minute
                sleep_time = 4 - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()

        try:
            config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }

            if stream:
                result_text = ""
                for chunk in self.client.models.generate_content_stream(
                    model=self.model,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    generation_config=config,
                ):
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        result_text += chunk.text
                print()
                return result_text
            else:
                response = self.client.models.generate_content(
                model=self.model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config=config,   
            )
                text = getattr(response, "text", "").strip()
                return text
        except Exception as e:
            print("Error from Gemini:", e)
            return ""
