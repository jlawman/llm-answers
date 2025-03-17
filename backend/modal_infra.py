from modal import Image, App, asgi_app, Secret
from pathlib import Path

# Create a stub for the Modal application
app = App("llml-answers")

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
def ask_llm(prompt: str, model_type: str, provider: str, model_name: str, system_prompt: str, use_thinking: bool, max_tokens: int, xml_tags: List[str], xml_outer_tag: str):
    from main import llm
    return llm(prompt, model_type, provider, model_name, system_prompt, use_thinking, max_tokens, xml_tags, xml_outer_tag)

# Add a local entrypoint so you can run with `modal run`
if __name__ == "__main__":
    app.serve()
