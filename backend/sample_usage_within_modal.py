import modal

# Create a minimal app
app = modal.App("simple-llm-client")

# Import the ask_llm function from the llm-answers app
ask_llm_function = modal.Function.from_name("llm-answers", "ask_llm")
#https://modal.com/docs/reference/modal.Function


@app.local_entrypoint()
def main():
    # Call the imported function with a simple prompt
    result = ask_llm_function.remote(prompt="What 2+2") #Remove can be removed if running modal.
    
    # Print the response
    print("\n=== LLM Response ===")
    print(result["response"])
    
    # Print some metadata
    print("\n=== Metadata ===")
    print(f"Model: {result['provider']}/{result['model']}")
    print(f"Token usage: {result['usage']['total_tokens']}")

if __name__ == "__main__":
    # Run with: modal run backend/simple_llm_client.py
    main()