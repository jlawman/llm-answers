import re
import os

def ask_anthropic(prompt: str, model: str, use_thinking: bool = False, system_prompt: str = None, max_tokens: int = 1000):
    """
    Send prompt to Anthropic.
    """
    from anthropic import Anthropic
    messages = [{"role": "user", "content": prompt}]

    client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
    
    print(f"Messages: {messages}")

    # Using streaming for both request types to avoid timeout issues
    if use_thinking:
        if system_prompt:
            response_stream = client.beta.messages.create(
                model=model,
                max_tokens=max_tokens,
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
                max_tokens=max_tokens,
                system=system_prompt,
                stream=True  # Enable streaming
            )
        else:
            response_stream = client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
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

def ask_groq(prompt: str, model: str, system_prompt: str = None, use_thinking: bool = False, max_tokens: int = 1000):
    """
    Send prompt to Groq.
    """
    from groq import Groq

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

def ask_google(prompt: str, model: str, system_prompt: str = None, use_thinking: bool = False, max_tokens: int = 1000):
    from google.genai import types
    from google import genai


    client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

    if use_thinking:
        raise HTTPException(status_code=500, detail="No support for thinking in Google")

    if system_prompt:
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                maxOutputTokens=max_tokens),
            contents=prompt)
    else:
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                maxOutputTokens=max_tokens),
            contents=prompt)
    return response.text, ""


def ask_openai(prompt: str, model: str, system_prompt: str = None, use_thinking: bool = False, max_tokens: int = 1000):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    messages = [{"role": "user", "content": prompt}]

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )

    if use_thinking:
        response = response.choices[0].message.content
        thinking = "redacted"
    else:
        response = response.choices[0].message.content
        thinking = ""

    return response, thinking
