from fastapi import FastAPI, HTTPException
from anthropic import Anthropic
import json
import random
import os
from groq import Groq
import re
from google import genai

app = FastAPI(
    title="LLM Answers API",
        description="""
        This API provides endpoints for generating answers from LLMs:
    
    - `/v1/ask-default`: Ask the default model a question (Claude 3.7 Sonnet)
    - `/v1/ask-cheap`: Ask using a cost-effective model (Gemini)
    - `/v1/ask-thinking-fast`: Ask with visible reasoning using a fast model (Groq)
    - `/v1/ask-thinking`: Ask with visible reasoning (Anthropic Claude)
    - `/v1/ask-fast`: Ask using a fast model (Groq)
    
   
    """,
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

def ask_anthropic(prompt: str, model: str, thinking: bool = False, system_prompt: str = None):
    """
    Send prompt to Anthropic.
    """
    messages = [{"role": "user", "content": prompt}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    
    print(f"Messages: {messages}")

    # Using streaming for both request types to avoid timeout issues
    if thinking:
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
        return {"response": response_content, "thinking": thinking_content}
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
        
        return response_content

def ask_groq(prompt: str, model: str, system_prompt: str = None):
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
    return chat_completion.choices[0].message.content

def ask_google(prompt: str, model: str):
    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)
    return response.text

@app.get('/v1/ask-default')
async def ask_default(prompt: str):
    """_summary_
    Ask a high quality model.
    """
    try:
       response = ask_anthropic(prompt, "claude-3-7-sonnet-latest", thinking=False)
       return {
        "response": response
       }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get('/v1/ask-cheap')
async def ask_cheap(prompt: str):
    """_summary_
    Ask a cheap model.
    """
    try:
        response = ask_google(prompt, "gemini-2.0-flash")
        return {
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.get('/v1/ask-thinking-fast')
async def ask_thinking_fast(prompt: str):
    try:
        # Use Groq with thinking
        response_with_thinking = ask_groq(prompt, "deepseek-r1-distill-llama-70b")
        print(f"Response with thinking: {response_with_thinking}")
        # Extract thinking from XML tags if present
        thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        thinking_match = thinking_pattern.search(response_with_thinking)
        
        if thinking_match:
            thinking = thinking_match.group(1).strip()
            # Remove the thinking section from the response
            response = thinking_pattern.sub('', response_with_thinking).strip()
        else:
            thinking = ""
            response = response_with_thinking

        return {
                "response": response,
                "thinking": thinking
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get('/v1/ask-thinking')
async def ask_thinking(prompt: str):
    try:
        # Use Anthropic with thinking
        result = ask_anthropic(prompt, "claude-3-7-sonnet-20250219", thinking=True)
        
        # For thinking-enabled requests, result is already a dictionary
        # with "response" and "thinking" keys
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get('/v1/ask-fast')
async def ask_fast(prompt: str):
    try:
        # Use Groq for fast responses
        response = ask_groq(prompt, "llama-3.3-70b-versatile")
        
        return {
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
