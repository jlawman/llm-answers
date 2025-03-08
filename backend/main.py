from fastapi import FastAPI, HTTPException
from anthropic import Anthropic
import json
import random
import os
from groq import Groq
import re
from google import genai
from openai import OpenAI
from google.genai import types

app = FastAPI(
    title="LLM Answers API",
        description="""
        This API provides endpoints for generating answers from LLMs:
    
    - `/v1/ask-default`: Ask the default model a question
    - `/v1/ask-cheap`: Ask using a cost-effective model       
    - `/v1/ask-thinking-fast`: Ask with visible reasoning using a fast model
    - `/v1/ask-thinking`: Ask with visible reasoning
    - `/v1/ask-fast`: Ask using a fast model 
    
   
    """,
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

FAST_MODEL = ("groq", "llama-3.3-70b-versatile")
THINKING_FAST_MODEL = ("groq", "deepseek-r1-distill-llama-70b")
CHEAP_MODEL = ("google", "gemini-2.0-flash")
THINKING_MODEL = ("anthropic", "claude-3-7-sonnet-20250219")
DEFAULT_MODEL = ("anthropic", "claude-3-7-sonnet-latest")

def ask_anthropic(prompt: str, model: str, use_thinking: bool = False, system_prompt: str = None):
    """
    Send prompt to Anthropic.
    """
    messages = [{"role": "user", "content": prompt}]

    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    
    print(f"Messages: {messages}")

    # Using streaming for both request types to avoid timeout issues
    if use_thinking:
        if system_prompt:
            response_stream = client.beta.messages.create(
                model=model,
                max_tokens=128000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 32000
                },
                messages=messages,
                system=system_prompt,
                betas=["output-128k-2025-02-19"],
                stream=True  # Enable streaming
            )
        else:
            response_stream = client.beta.messages.create(
                model=model,
                max_tokens=128000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 32000
                },
                messages=messages,
                betas=["output-128k-2025-02-19"],
                stream=True  # Enable streaming
            )
        
        # Collect both thinking content and the actual response
        thinking_content = ""
        response_content = ""
        
        for chunk in response_stream:
            print(f"Chunk: {chunk}")
            
            # Extract thinking content from BetaThinkingDelta
            if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'thinking'):
                thinking_content += chunk.delta.thinking
            
            # Extract text content from BetaTextDelta
            elif hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                response_content += chunk.delta.text
                
            elif hasattr(chunk, 'type') and chunk.type == 'message_stop':
                break
        
        # Return both response and thinking separately
        return response_content, thinking_content
    else:
        if system_prompt:
            response_stream = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=10000,
                system=system_prompt,
                stream=True  # Enable streaming
            )
        else:
            response_stream = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=10000,
                stream=True  # Enable streaming
            )
        
        # For non-thinking responses, just collect the text
        response_content = ""
        for chunk in response_stream:
            print(f"Chunk: {chunk}")
            
            # Check if it's a content block delta with text
            if hasattr(chunk, 'type') and chunk.type == 'content_block_delta':
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    response_content += chunk.delta.text
            # Alternative check based on class name
            elif 'RawContentBlockDeltaEvent' in str(type(chunk)):
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    response_content += chunk.delta.text
            # Stop when the message is complete
            elif hasattr(chunk, 'type') and chunk.type == 'message_stop':
                break
        
        return response_content, ""

def ask_groq(prompt: str, model: str, system_prompt: str = None, use_thinking: bool = False):
    """
    Send prompt to Groq.
    """
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    messages = [{"role": "user", "content": prompt}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=10000
    )

    if use_thinking:
        thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        thinking_match = thinking_pattern.search(chat_completion.choices[0].message.content)
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # Remove the thinking section from the response
            response = thinking_pattern.sub('', chat_completion.choices[0].message.content).strip()
        else:
            thinking = ""
            response = chat_completion.choices[0].message.content
    else:
        response = chat_completion.choices[0].message.content
        thinking = ""
    return response, thinking

def ask_google(prompt: str, model: str, system_prompt: str = None, use_thinking: bool = False):
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    if use_thinking:
        raise HTTPException(status_code=500, detail="No support for thinking in Google")

    if system_prompt:
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt),
            contents=prompt)
    else:
        response = client.models.generate_content(
            model=model,
            contents=prompt)
    return response.text, ""


def ask_openai(prompt: str, model: str, system_prompt: str = None, use_thinking: bool = False):
    
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    messages = [{"role": "user", "content": prompt}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    if use_thinking:
        response = response.choices[0].message.content
        thinking = "redacted"
    else:
        response = response.choices[0].message.content
        thinking = ""

    return response, thinking


@app.get('/v1/ask-default')
async def ask_default(prompt: str, system_prompt: str = None):
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


@app.get('/v1/ask-cheap')
async def ask_cheap(prompt: str, system_prompt: str = None):
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


@app.get('/v1/ask-thinking-fast')
async def ask_thinking_fast(prompt: str, system_prompt: str = None):
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

@app.get('/v1/ask-thinking')
async def ask_thinking(prompt: str, system_prompt: str = None):
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

@app.get('/v1/ask-fast')
async def ask_fast(prompt: str):
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
