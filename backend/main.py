FAST_MODEL = ("groq", "llama-3.3-70b-versatile")
THINKING_FAST_MODEL = ("groq", "deepseek-r1-distill-llama-70b")
CHEAP_MODEL = ("google", "gemini-2.0-flash")
THINKING_MODEL = ("anthropic", "claude-3-7-sonnet-20250219")
DEFAULT_MODEL = ("anthropic", "claude-3-7-sonnet-latest")
########################################################


import os
import time
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any

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

def parse_xml_response(response: str, xml_tags: Optional[List[str]] = None, xml_outer_tag: Optional[str] = None) -> Dict[str, str]:
    """
    Parse the XML response based on the specified tags.
    
    Args:
        response: The raw response from the LLM
        xml_tags: List of XML tags to extract content from
        xml_outer_tag: Optional outer wrapper tag
        
    Returns:
        Dictionary with tag names as keys and their content as values
    """
    parsed_response = {}
    
    # If no tags specified, return empty dict
    if not xml_tags:
        return parsed_response
    
    # Case 1: With outer tag and inner tags
    if xml_outer_tag:
        outer_pattern = f"<{xml_outer_tag}>(.*?)</{xml_outer_tag}>"
        outer_match = re.search(outer_pattern, response, re.DOTALL)
        
        if outer_match:
            outer_content = outer_match.group(1)
            for tag in xml_tags:
                pattern = f"<{tag}>(.*?)</{tag}>"
                match = re.search(pattern, outer_content, re.DOTALL)
                if match:
                    parsed_response[tag] = match.group(1).strip()
                else:
                    parsed_response[tag] = ""
    
    # Case 2 & 3: No outer tag or outer tag not found
    if not xml_outer_tag or not outer_match:
        for tag in xml_tags:
            pattern = f"<{tag}>(.*?)</{tag}>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                parsed_response[tag] = match.group(1).strip()
            else:
                parsed_response[tag] = ""
    
    return parsed_response

def estimate_token_usage(prompt: str, response: str, provider: str) -> Dict[str, int]:
    """
    Estimate token usage for the request and response.
    
    Args:
        prompt: The input prompt
        response: The model's response
        provider: The LLM provider (anthropic, google, groq, openai)
        
    Returns:
        Dictionary with token usage statistics
    """
    # Simple estimation based on words (not perfect but gives a rough estimate)
    # Different providers have different tokenization methods
    # For a production system, you might want to use provider-specific tokenizers
    
    def estimate_tokens(text: str) -> int:
        # Rough estimation: ~4 characters per token on average
        return len(text) // 4
    
    prompt_tokens = estimate_tokens(prompt)
    completion_tokens = estimate_tokens(response)
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

@app.get('/v1/llm')#, token: HTTPAuthorizationCredentials = Depends(verify_token)
async def ask_llm(prompt: str
                  , system_prompt: str = None
                  , model_type: str = None
                  , provider: str = None
                  , model_name: str = None
                  , use_thinking: bool = False
                  , max_tokens: int = 1000
                  #, project_name: str = "red-panda-simple"
                  , xml_tags: list[str] = None
                  , xml_outer_tag: str = None):
    """_summary_
    REQUIRED PARAMETERS:
    - prompt: string       // The query to send to the model

    OPTIONAL PARAMETERS:
    - model_type: string        // "default", "default-thinking", "fast", "fast-thinking", "cheap"
    - provider: string          // "anthropic", "google", "groq", etc. (overrides model_type)
    - model_name: string        // Specific model name (used with provider)
    - system_prompt: string     // Custom system instructions
    - use_thinking: boolean     // Force thinking mode (default: false, auto-true for *-thinking types)
    - temperature: float        // Control randomness (default: 0.7)
    - max_tokens: integer       // Limit response length
    - project_name: str // Project name for braintrust logging. Defaults to red-panda-simple

    XML FORMATTING (default):
    - xml_tags: string[]        // List of tags for response structure ["reasoning", "answer"]
    - xml_outer_tag: string     // Optional wrapper tag (e.g., "response")

    RESPONSE OBJECT:
    {
    response: string,         // Raw model response text
    parsed_response: {        // Structured data extracted from XML tags
        [tag_name]: string      // Content from each requested tag
    },
    thinking: string,         // Reasoning output (if enabled)
    use_thinking: boolean,    // Whether thinking was used
    provider: string,         // Provider used for this request
    model: string,            // Specific model used
    usage: {                  // Token usage statistics
        prompt_tokens: integer,
        completion_tokens: integer,
        total_tokens: integer
        }
    }

    """

    if provider and model_name:
        print(f"Provider ({provider}) and model name ({model_name}) provided:")
    elif model_type is None and (provider is None or model_name is None) or model_type == "default":
        provider, model_name = DEFAULT_MODEL
        use_thinking = False
    elif model_type == "default-thinking":
        provider, model_name = THINKING_MODEL
        use_thinking = True
    elif model_type == "fast":
        provider, model_name = FAST_MODEL
        use_thinking = True
    elif model_type == "fast-thinking":
        provider, model_name = THINKING_FAST_MODEL
        use_thinking = True
    elif model_type == "cheap":
        provider, model_name = CHEAP_MODEL
        use_thinking = False
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {model_type}, with no provider ({provider}) or model_name ({model_name}) provided")

    try:
        if provider == "anthropic":
            response, thinking = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model_name, use_thinking=use_thinking, max_tokens=max_tokens)
        elif provider == "google":
            response, thinking = ask_google(system_prompt=system_prompt, prompt=prompt, model=model_name, use_thinking=use_thinking, max_tokens=max_tokens)
        elif provider == "groq":
            response, thinking = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model_name, use_thinking=use_thinking, max_tokens=max_tokens)
        elif provider == "openai":
            response, thinking = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model_name, use_thinking=use_thinking, max_tokens=max_tokens)
        else:
            raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
        # Parse the response based on XML tags
        parsed_response = parse_xml_response(response, xml_tags, xml_outer_tag)
        
        # Estimate token usage
        usage = estimate_token_usage(prompt, response, provider)

        return {
            "response": response,
            "parsed_response": parsed_response,
            "use_thinking": use_thinking,
            "thinking": thinking,
            "provider": provider,
            "model": model_name,
            "usage": usage
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")



# @app.get('/v1/llm/default')
# async def ask_default(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     """_summary_
#     Ask a high quality model.
#     """
#     provider, model = DEFAULT_MODEL
#     use_thinking = False
#     try:
#         if provider == "anthropic":
#             response, _ = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "google":
#             response, _ = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "groq":
#             response, _ = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "openai":
#             response, _ = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         else:
#             raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
#         return {
#             "response": response,
#             "use_thinking": use_thinking,
#             "thinking": "",
#             "provider": provider,
#             "model": model
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# @app.get('/v1/llm/cheap')
# async def ask_cheap(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     """_summary_
#     Ask a cheap model.
#     """
#     provider, model = CHEAP_MODEL
#     use_thinking = False
#     try:
#         if provider == "google":
#             response, _ = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "groq":
#             response, _ = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "openai":
#             response, _ = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "anthropic":
#             response, _ = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         else:
#             raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
#         return {
#             "response": response,
#             "use_thinking": use_thinking,
#             "thinking": "",
#             "provider": provider,
#             "model": model
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


# @app.get('/v1/llm/fast-thinking')
# async def ask_thinking_fast(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     provider, model = THINKING_FAST_MODEL
#     use_thinking = True
#     try:
#         # Use Groq with thinking
#         if provider == "groq":
#             response, thinking = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "anthropic":
#             response, thinking = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "google":
#             response, thinking = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "openai":
#             response, thinking = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         else:
#             raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
#         return {
#             "response": response,
#             "use_thinking": True,
#             "thinking": thinking,
#             "provider": provider,
#             "model": model
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# @app.get('/v1/llm/default-thinking')
# async def ask_thinking(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     provider, model = THINKING_MODEL
#     use_thinking = True
#     try:
#         # Use Anthropic with thinking
#         if provider == "anthropic":
#             response, thinking = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "google":
#             response, thinking = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "groq":
#             response, thinking = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "openai":
#             response, thinking = ask_openai(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         else:
#             raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
#         return {
#             "response": response,
#             "use_thinking": use_thinking,
#             "thinking": thinking,
#             "provider": provider,
#             "model": model
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# @app.get('/v1/llm/fast')
# async def ask_fast(prompt: str, system_prompt: str = None, token: HTTPAuthorizationCredentials = Depends(verify_token)):
#     provider, model = FAST_MODEL
#     use_thinking = False
#     try:
#         # Use Groq for fast responses
#         if provider == "groq":
#             response, _ = ask_groq(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "anthropic":
#             response, _ = ask_anthropic(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "google":
#             response, _ = ask_google(system_prompt=system_prompt, prompt=prompt, model=model, use_thinking=use_thinking)
#         elif provider == "openai":
#             response, _ = ask_openai(system_prompt=system_prompt, prompt=prompt, model= model, use_thinking=use_thinking)
#         else:
#             raise HTTPException(status_code=500, detail=f"Invalid model: {provider}")
        
#         return {
#             "response": response,
#             "use_thinking": False,
#             "thinking": "",
#             "provider": provider,
#             "model": model
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
