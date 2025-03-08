'use client'

import React, { useState } from 'react'
import { Container } from '@/components/ui/container'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Loader2 } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'

// Add these type definitions at the top of the file
type LLMResponse = {
  content?: string;
  error?: string;
  [key: string]: string | undefined;
};

type LoadingState = Record<string, boolean>;

export default function LLMDemo() {
  const [message, setMessage] = useState('')
  const [anthropicResponse, setAnthropicResponse] = useState<LLMResponse | null>(null)
  const [openAIResponse, setOpenAIResponse] = useState<LLMResponse | null>(null)
  const [groqResponse, setGroqResponse] = useState<LLMResponse | null>(null)
  const [loading, setLoading] = useState<LoadingState>({})

  const askLLM = async (
    endpoint: string,
    setResponse: React.Dispatch<React.SetStateAction<LLMResponse | null>>,
    model: string
  ) => {
    const loadingKey = `${endpoint}-${model}`
    setLoading(prev => ({ ...prev, [loadingKey]: true }))
    try {
      const response = await fetch(`/api/ask-${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, model }),
      })
      const data = await response.json()
      console.log(`${endpoint} API response:`, data)
      setResponse(data)
    } catch (error) {
      console.error(`Error calling ${endpoint} API:`, error)
    } finally {
      setLoading(prev => ({ ...prev, [loadingKey]: false }))
    }
  }

  const providers = [
    {
      name: 'Anthropic',
      action: (model: string) => askLLM('anthropic', setAnthropicResponse, model),
      response: anthropicResponse,
      models: ['claude-3-5-sonnet-20240620','claude-3-opus-20240229','claude-3-haiku-20240307'],
    },
    {
      name: 'OpenAI',
      action: (model: string) => askLLM('openai', setOpenAIResponse, model),
      response: openAIResponse,
      models: ['gpt-4o', 'gpt-4o-mini', 'o1-mini'],
    },
    {
      name: 'Groq',
      action: (model: string) => askLLM('groq', setGroqResponse, model),
      response: groqResponse,
      models: ['llama-3.2-90b-text-preview', 'llama-3.2-11b-text-preview'],
    },
  ]

  return (
    <Container className="py-16">
      <h1 className="text-4xl font-bold mb-8">LLM API Demo</h1>
      
      <div className="mb-8">
        <Input
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter your message here"
          className="mb-4"
        />
        <div className="flex flex-wrap gap-4 mb-4">
          {providers.map((provider) => (
            <div key={provider.name} className="flex-1 min-w-[200px]">
              <h3 className="text-lg font-semibold mb-2">{provider.name}</h3>
              <div className="flex flex-col gap-2">
                {provider.models.map((model) => {
                  const loadingKey = `${provider.name.toLowerCase()}-${model}`
                  return (
                    <Button
                      key={model}
                      onClick={() => provider.action(model)}
                      disabled={loading[loadingKey]}
                      className="w-full text-sm"
                      title={`${provider.name}: ${model}`}
                    >
                      {loading[loadingKey] ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                      {model}
                    </Button>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-8">
        {providers.map((provider) => (
          <div key={provider.name} className="border rounded-lg p-4">
            <h2 className="text-2xl font-bold mb-4">{provider.name} Response:</h2>
            {provider.response && (
              <ScrollArea className="h-64 w-full rounded-md border p-4">
                <div className="whitespace-pre-wrap">{JSON.stringify(provider.response, null, 2)}</div>
              </ScrollArea>
            )}
          </div>
        ))}
      </div>
    </Container>
  )
}
