FAST_MODEL = ("groq", "llama-3.3-70b-versatile")
THINKING_FAST_MODEL = ("groq", "deepseek-r1-distill-llama-70b")
CHEAP_MODEL = ("google", "gemini-2.0-flash")
THINKING_MODEL = ("anthropic", "claude-3-7-sonnet-20250219")
DEFAULT_MODEL = ("anthropic", "claude-3-7-sonnet-latest")
########################################################


import os
import time
import hashlib

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from utils.ask_llms import ask_anthropic, ask_google, ask_groq, ask_openai


app = FastAPI(
    title="LLM Answers API",
        description="""
        This API provides endpoints for generating answers from LLMs:
    
    - `/v1/llm/default`: Ask the default model a question
    - `/v1/llm/default-thinking`: Ask with visible reasoning
    - `/v1/llm/cheap`: Ask using a cost-effective model       
    - `/v1/llm/fast`: Ask using a fast model 
    - `/v1/llm/fast-thinking`: Ask with visible reasoning using a fast model
    
   
    """,
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

request_tracker = {}

    
auth_scheme = HTTPBearer()

async def verify_token(token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    # Authentication using the standard approach
    if token.credentials != os.environ["ENDPOINT_API_KEY"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized access. Please provide a valid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Rate limiting logic
    client_id = hashlib.md5(str(token.credentials).encode()).hexdigest()
    current_time = time.time()
    
    if client_id in request_tracker:
        last_request_time, count = request_tracker[client_id]
        # Allow 5 requests per minute
        if current_time - last_request_time < 60:
            if count >= 5:
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limit exceeded. Try again later."
                )
            request_tracker[client_id] = (last_request_time, count + 1)
        else:
            request_tracker[client_id] = (current_time, 1)
    else:
        request_tracker[client_id] = (current_time, 1)
    
    return token






@app.get('/v1/llm/default')
async def ask_default(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    """_summary_
    Ask a high quality model.
    """
    provider, model = DEFAULT_MODEL
    use_thinking = False
    try:
        if provider == "anthropic":
            response, _ = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "google":
            response, _ = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "groq":
            response, _ = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "openai":
            response, _ = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        return {
            "response": response,
            "use_thinking": use_thinking,
            "thinking": "",
            "provider": provider,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get('/v1/llm/cheap')
async def ask_cheap(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    """_summary_
    Ask a cheap model.
    """
    provider, model = CHEAP_MODEL
    use_thinking = False
    try:
        if provider == "google":
            response, _ = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "groq":
            response, _ = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "openai":
            response, _ = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "anthropic":
            response, _ = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        return {
            "response": response,
            "use_thinking": use_thinking,
            "thinking": "",
            "provider": provider,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get('/v1/llm/fast-thinking')
async def ask_thinking_fast(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    provider, model = THINKING_FAST_MODEL
    use_thinking = True
    try:
        # Use Groq with thinking
        if provider == "groq":
            response, thinking = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "anthropic":
            response, thinking = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "google":
            response, thinking = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "openai":
            response, thinking = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
        return {
            "response": response,
            "use_thinking": True,
            "thinking": thinking,
            "provider": provider,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get('/v1/llm/default-thinking')
async def ask_thinking(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    provider, model = THINKING_MODEL
    use_thinking = True
    try:
        # Use Anthropic with thinking
        if provider == "anthropic":
            response, thinking = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "google":
            response, thinking = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "groq":
            response, thinking = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        elif provider == "openai":
            response, thinking = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
        return {
            "response": response,
            "use_thinking": use_thinking,
            "thinking": thinking,
            "provider": provider,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get('/v1/llm/fast')
async def ask_fast(prompt: str, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    provider, model = FAST_MODEL
    try:
        # Use Groq for fast responses
        if provider == "groq":
            response, _ = ask_groq(prompt, model)
        elif provider == "anthropic":
            response, _ = ask_anthropic(prompt, model)
        elif provider == "google":
            response, _ = ask_google(prompt, model)
        elif provider == "openai":
            response, _ = ask_openai(prompt, model)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
        return {
            "response": response,
            "use_thinking": False,
            "thinking": "",
            "provider": provider,
            "model": model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
