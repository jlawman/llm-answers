# General LLM Endpoint and Function within Modal

A simple API service that provides unified access to multiple LLM providers (Anthropic, Google, Groq, OpenAI) through a single endpoint, deployed using Modal. 

This is intended to be a convenience function for quick access to multiple LLMs and to handle some XML parsing on the output. I also wanted to give quick LLM access to various modal.com projects while adding minimal complexity.


## Getting Started

### Prerequisites

- A free Modal.com account and CLI set up
- API keys for the LLM providers you want to use

### Setup

1. Clone this repository
2. Set up your Modal secrets:
   ```
   modal secret create llms \
     ANTHROPIC_API_KEY=your_anthropic_key \
     OPENAI_API_KEY=your_openai_key \
     GROQ_API_KEY=your_groq_key \
     GOOGLE_API_KEY=your_google_key
   ```

3. Set up API security:
   ```
   modal secret create API_SECURITY ENDPOINT_API_KEY=your_chosen_api_key
   ```

4. Deploy to Modal:
   ```
   modal deploy modal_infra.py
   ```

## Usage

### REST API

Make POST requests to the `/v1/llm` endpoint:

```python
import requests

response = requests.post(
    "https://your-modal-endpoint/v1/llm",
    headers={"Authorization": f"Bearer {your_api_key}"},
    json={
        "prompt": "What are the key features of Python? Return your response in <reasoning> and <answer> tags nested within a <response> tag.",
        "system_prompt": "You are a helpful programming assistant.",
        "model_type": "default",  # Options: default, default-thinking, fast, fast-thinking, cheap
        "use_thinking": False,
        "max_tokens": 1000,
        "xml_tags": ["reasoning", "answer"], #Optional, only used if xml_outer_tag is provided in the prompt response.
        "xml_outer_tag": "response" #Optional, only used if xml_tags are provided in output.
    }
)

print(response.json())
```

### Response Object

```json
{
    "response": "string",       
    "parsed_response": {        
        "[tag_name_1]": "string",
        "[tag_name_2]": "string",
        "[tag_name_N]": "string"  
    },
    "thinking": "string",       
    "use_thinking": true,
    "provider": "string",        
    "model": "string",            
    "usage": {                 
        "prompt_tokens": 123,
        "completion_tokens": 123,
        "total_tokens": 123
    }
}
```

Note: the usage values are just a ballpark estimate based on the prompt and response length. Provider values or tiktoken could be added for more accurate estimates.

### Modal Function

Use the `ask_llm` function directly in your other Modal apps:

```python
import modal

app = modal.App("my-app")

def my_function():
    ask_llm_function = modal.Function.from_name("llm-answers", "ask_llm")
    result = ask_llm_function.remote(
        prompt="What is 2+2?"
    )
    return result
```

### Sample Deployment and Usage from within Modal

First run:
modal deploy modal_infra.py

Then run:
modal run sample_usage_within_modal.py 

## Available Models

Models will change, but here are the current options:

- **Default**: Anthropic Claude 3.7 Sonnet
- **Default-Thinking**: Anthropic Claude 3.7 Sonnet with thinking mode
- **Fast**: Groq Llama-3.3-70b-versatile
- **Fast-Thinking**: Groq DeepSeek R1 Distill Llama 70B with thinking mode
- **Cheap**: Google Gemini 2.0 Flash

You can also specify a provider and model name directly instead of using the predefined types.


## Features not currently supported

Following features are not currently supported since I do not currently need them for my intended use case of this project.

- Forced JSON output
- Complex XML structures
- Streaming responses
- Complex message structures
- Passing images or files to the LLM
- Function calling
- Braintrust logging