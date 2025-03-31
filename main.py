FAST_MODEL = ("groq", "llama-3.3-70b-versatile")
THINKING_FAST_MODEL = ("groq", "deepseek-r1-distill-llama-70b")
CHEAP_MODEL = ("google", "gemini-2.0-flash")
THINKING_MODEL = ("anthropic", "claude-3-7-sonnet-latest")
DEFAULT_MODEL = ("anthropic", "claude-3-7-sonnet-latest")
########################################################


import os
import time
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException, Depends, status, Body, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import hmac
import secrets
from datetime import datetime, timedelta

from utils.ask_llms import ask_anthropic, ask_google, ask_groq, ask_openai


app = FastAPI(
    title="LLM Answers API",
    version="1.0.0",
    docs_url="/",
    redoc_url="/redoc"
)

# Use a more persistent structure with expiry
class RateLimitStore:
    def __init__(self):
        self.store = {}
        self.last_cleanup = datetime.now()
    
    def add_request(self, client_id, ip_address):
        current_time = datetime.now()
        
        # Clean up old entries every 10 minutes
        if (current_time - self.last_cleanup).total_seconds() > 600:
            self._cleanup()
            self.last_cleanup = current_time
        
        # Create composite key from client_id and IP
        composite_key = f"{client_id}:{ip_address}"
        
        if composite_key in self.store:
            requests = self.store[composite_key]
            # Remove requests older than 1 minute
            current_window = [t for t in requests if (current_time - t).total_seconds() < 60]
            current_window.append(current_time)
            self.store[composite_key] = current_window
            return len(current_window)
        else:
            self.store[composite_key] = [current_time]
            return 1
    
    def _cleanup(self):
        current_time = datetime.now()
        for key in list(self.store.keys()):
            self.store[key] = [t for t in self.store[key] if (current_time - t).total_seconds() < 60]
            if not self.store[key]:
                del self.store[key]

request_tracker = RateLimitStore()

# Use constant-time comparison for API key validation
def secure_compare(a, b):
    return hmac.compare_digest(a, b)

auth_scheme = HTTPBearer()

async def verify_token(request: Request, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    # Authentication using constant-time comparison to prevent timing attacks
    if not secure_compare(token.credentials, os.environ["ENDPOINT_API_KEY"]):
        # Use generic error message
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Rate limiting with IP + API key
    client_id = hashlib.md5(str(token.credentials).encode()).hexdigest()
    ip_address = request.client.host
    
    # Get request count for this client/IP combination
    request_count = request_tracker.add_request(client_id, ip_address)
    
    # Allow 20 requests per minute
    if request_count > 20:
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Try again later."
        )
    
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

class LLMRequest(BaseModel):
    prompt: str = Field(..., description="The query to send to the model", example="What are the key features of Python?")
    system_prompt: Optional[str] = Field(None, description="Custom system instructions", example="You are a helpful programming assistant specializing in Python.")
    model_type: Optional[str] = Field(None, description="Model type to use", example="default")
    provider: Optional[str] = Field(None, description="LLM provider name", example="anthropic")
    model_name: Optional[str] = Field(None, description="Specific model name", example="claude-3-7-sonnet-latest")
    use_thinking: bool = Field(False, description="Whether to show reasoning process")
    max_tokens: int = Field(1000, description="Maximum tokens in response", ge=1, le=1_000_000)
    xml_tags: Optional[List[str]] = Field(None, description="Expected XML tags for structured response", example=["reasoning", "answer"])
    xml_outer_tag: Optional[str] = Field(None, description="Expected outer wrapper XML tag", example="response")
    
    # Add validators for security
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Prompt cannot be empty")
        if len(v) > 250000:  # Reasonable limit
            raise ValueError("Prompt exceeds maximum length")
        return v
    
    @validator('system_prompt')
    def validate_system_prompt(cls, v):
        if v is not None and len(v) > 250000:  # Reasonable limit
            raise ValueError("System prompt exceeds maximum length")
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v is not None:
            valid_types = ["default", "default-thinking", "fast", "fast-thinking", "cheap"]
            if v not in valid_types:
                raise ValueError(f"Invalid model type. Must be one of: {', '.join(valid_types)}")
        return v
    
    @validator('provider')
    def validate_provider(cls, v):
        if v is not None:
            valid_providers = ["anthropic", "google", "groq", "openai"]
            if v not in valid_providers:
                raise ValueError(f"Invalid provider. Must be one of: {', '.join(valid_providers)}")
        return v
    
    @validator('xml_tags')
    def validate_xml_tags(cls, v):
        if v is not None:
            for tag in v:
                if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                    raise ValueError(f"Invalid XML tag: {tag}. Tags must contain only alphanumeric characters, underscores, and hyphens.")
        return v
    
    @validator('xml_outer_tag')
    def validate_xml_outer_tag(cls, v):
        if v is not None and not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError(f"Invalid XML outer tag: {v}. Tags must contain only alphanumeric characters, underscores, and hyphens.")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "What are the key features of Python?",
                "system_prompt": "You are a helpful programming assistant specializing in Python.",
                "model_type": "default",
                "use_thinking": True,
                "max_tokens": 1000,
                "xml_tags": ["reasoning", "answer"],
                "xml_outer_tag": "response"
            }
        }

@app.post('/v1/llm', dependencies=[Depends(verify_token)])
async def llm(request: LLMRequest):
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
    #- project_name: str // Project name for braintrust logging. Defaults to red-panda-simple

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
    try:
        return _process_llm_request(request)
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the actual error but return a generic message
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")


def _process_llm_request(request: LLMRequest):
    if request.provider and request.model_name:
        print(f"Provider ({request.provider}) and model name ({request.model_name}) provided:")
    elif request.model_type is None and (request.provider is None or request.model_name is None) or request.model_type == "default":
        request.provider, request.model_name = DEFAULT_MODEL
        request.use_thinking = False
    elif request.model_type == "default-thinking":
        print("Using default-thinking model")
        request.provider, request.model_name = THINKING_MODEL
        request.use_thinking = True
    elif request.model_type == "fast":
        request.provider, request.model_name = FAST_MODEL
        request.use_thinking = False
    elif request.model_type == "fast-thinking":
        request.provider, request.model_name = THINKING_FAST_MODEL
        request.use_thinking = True
    elif request.model_type == "cheap":
        request.provider, request.model_name = CHEAP_MODEL
        request.use_thinking = False
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model type: {request.model_type}, with no provider ({request.provider}) or model_name ({request.model_name}) provided")

    try:
        if request.provider == "anthropic":
            response, thinking = ask_anthropic(system_prompt=request.system_prompt, prompt=request.prompt, model=request.model_name, use_thinking=request.use_thinking, max_tokens=request.max_tokens)
        elif request.provider == "google":
            response, thinking = ask_google(system_prompt=request.system_prompt, prompt=request.prompt, model=request.model_name, use_thinking=request.use_thinking, max_tokens=request.max_tokens)
        elif request.provider == "groq":
            response, thinking = ask_groq(system_prompt=request.system_prompt, prompt=request.prompt, model=request.model_name, use_thinking=request.use_thinking, max_tokens=request.max_tokens)
        elif request.provider == "openai":
            response, thinking = ask_openai(system_prompt=request.system_prompt, prompt=request.prompt, model=request.model_name, use_thinking=request.use_thinking, max_tokens=request.max_tokens)
        else:
            # Generic error message
            raise HTTPException(status_code=400, detail="Invalid provider specified")
        
        # Parse the response based on XML tags
        parsed_response = parse_xml_response(response, request.xml_tags, request.xml_outer_tag)
        
        # Estimate token usage
        usage = estimate_token_usage(request.prompt, response, request.provider)

        return {
            "response": response,
            "parsed_response": parsed_response,
            "use_thinking": request.use_thinking,
            "thinking": thinking,
            "provider": request.provider,
            "model": request.model_name,
            "usage": usage
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as they already have appropriate error details
        raise
    except Exception as e:
        # Log the actual error for debugging but return a generic message
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal server error occurred")



# New endpoint to run 3 LLM queries in parallel
@app.post('/v1/llm/parallel')
async def run_parallel_llms(base_page: str = Body(...), directions: str = Body(...)):
    """
    Runs three LLM queries in parallel using the provided inputs.
    
    Args:
        base_page: A string prompt that serves as the base page content
        directions: A string prompt that provides directions for processing
    """
    import time
    from modal_infra import ask_llm



    prompts = [
        f"""<base_page>{base_page}</base_page> 

<instructions>Wow let's make this Next.js 14 app router page look better and have a much improved UX experience! Let's also improve it by creating beautiful components that power it. Use literally any library you want!

Get inspired by TailwindUI, Sonner, Linear, and other people who care about making incredible UX.

Important however, just focus on the look and feel. You can do things like add additional buttons (or nav bar or sections) if the links exist elsewhere on the page. And you can possibly add a tiny bit of text if it makes sense. But DO NOT assume anything that can't be confidently inferred from the original page. (I.e. don't fill in missing text just because it should be there if you don't know what it is.). Don't add explainers of what the page or site does unless you are 200% sure.

Now let's make the page more pleasurable to use

Place your response in the following HTML format.

<response>
    <html>
        <body>
</instructions>""",
        f"""<base_page>{base_page}</base_page> 


<instructions>Wow let's make this Next.js 14 app router page look better and have a much improved UX experience! Let's also improve it by creating beautiful components that power it. Use literally any library you want!

Get inspired by TailwindUI, Sonner, Linear, and other people who care about making incredible UX.

Make this have immense amount of character.

Everything should be included in one file (even if the original page didn't have that). Place your response in the following HTML format.

<response>
    <html>
        <body>
            <h1>Hello World</h1>
</instructions>""",
        f"""<base_page>{base_page}</base_page> 

<instructions>Wow let's make this Next.js 14 app router page look better and have a much improved UX experience! Let's also improve it by creating beautiful components that power it. Use literally any library you want! But let's make it all one page.

Get inspired by TailwindUI and other people who care about making incredible UX.

Now let's make the page more delightful to use

Everything should be included in one file (even if the original page didn't have that). 

Place your response in the following HTML format.

<response>
    <html>
        <body>
            <h1>Hello World</h1>
        </body>
    </html>
</response>

</instructions>"""
    ]

    # Create inputs as a list of kwargs dictionaries
    inputs = prompts[:2] #[{"prompt": p, "model_type": "cheap", "xml_outer_tag": "response"} for p in prompts]

    print(f"Running {len(inputs)} LLM queries in parallel...")
    start_time = time.time()


    #inputs = ["Hi", "Hello"]
    # Use async for loop to consume the generator
    results_generator = ask_llm.map(inputs)
    results_list = [result async for result in results_generator]


    end_time = time.time()
    processing_time = end_time - start_time

    return {
        "processing_time": f"{processing_time:.2f} seconds",
        "num_queries": len(inputs),
        "results": results_list,
        "base_page": base_page[:100] + "..." if len(base_page) > 100 else base_page,
        "directions": directions[:100] + "..." if len(directions) > 100 else directions
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
