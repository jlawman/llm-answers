from modal import Image, App, asgi_app, Secret
from pathlib import Path

# Create a stub for the Modal application
app = App("llml-answers")

# Create an image with Python dependencies
image = (
    Image.debian_slim()
    .pip_install(["fastapi", "uvicorn", "anthropic", "google-genai", "groq"])
    .add_local_file("main.py", remote_path="/root/main.py")  # Mount main.py
)

# Define the ASGI app (changed from wsgi_app)
@app.function(image=image, secrets=[Secret.from_name("llms"), Secret.from_name("API_SECURITY")])
@asgi_app()
def fastapi_app():
    # Import FastAPI app from main.py
    from main import app
    return app

# Add a local entrypoint so you can run with `modal run`
if __name__ == "__main__":
    app.serve()
