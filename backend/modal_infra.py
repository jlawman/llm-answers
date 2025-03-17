from modal import Image, App, asgi_app, Secret
from pathlib import Path
from typing import List, Optional

app = App("llm-answers")

# Create an image with Python dependencies
image = (
    Image.debian_slim()
    .pip_install(["fastapi", "uvicorn", "openai","anthropic", "google-genai", "groq"])
    .add_local_file("main.py", remote_path="/root/main.py")  # Mount main.py
    .add_local_file("utils/ask_llms.py", remote_path="/root/utils/ask_llms.py")  # Mount utils
)

# Define the ASGI app (changed from wsgi_app)
@app.function(image=image, secrets=[Secret.from_name("llms"), Secret.from_name("API_SECURITY")])
@asgi_app()
def fastapi_app():
    # Import FastAPI app from main.py
    from main import app
    return app


### Coding agent, please leave this separate function alone.
@app.function(image=image, secrets=[Secret.from_name("llms")])
def ask_llm(prompt: str, model_type: str = None, provider: str = None, model_name: str = None, 
           system_prompt: str = None, use_thinking: bool = False, max_tokens: int = 1000, 
           xml_tags: List[str] = None, xml_outer_tag: str = None):
    from main import LLMRequest, _process_llm_request
    
    # Create a LLMRequest object
    request = LLMRequest(
        prompt=prompt,
        model_type=model_type,
        provider=provider,
        model_name=model_name,
        system_prompt=system_prompt,
        use_thinking=use_thinking,
        max_tokens=max_tokens,
        xml_tags=xml_tags,
        xml_outer_tag=xml_outer_tag
    )
    
    # Call the internal processing function directly
    return _process_llm_request(request)
