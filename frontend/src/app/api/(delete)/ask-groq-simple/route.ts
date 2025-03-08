import { NextRequest, NextResponse } from 'next/server';
import Groq from "groq-sdk";
import { logger } from '@/lib/braintrustlogger';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const { message, model = "llama-3.2-90b-text-preview" } = await request.json();

    if (!message) {
      return NextResponse.json({ error: 'No message provided' }, { status: 400 });
    }

    const result = await processGroqRequest(message, model);

    // Log asynchronously without waiting
    logger.traced(async (span) => {
      span.log({
        input: { message, model },
        output: result,
        tags: ['groq', 'request', 'response']
      });
    }, { name: 'ask_groq' }).catch(console.error);

    return NextResponse.json(result);
  } catch (error) {
    console.error('Error processing request:', error);
    
    // Log error asynchronously
    logger.traced(async (span) => {
      span.log({
        error: error instanceof Error ? error.message : 'Unknown error',
        tags: ['error', error instanceof Error ? 'known' : 'unknown']
      });
    }, { name: 'ask_groq_error' }).catch(console.error);

    if (error instanceof Error) {
      return NextResponse.json({ error: `Error processing request: ${error.message}` }, { status: 500 });
    } else {
      return NextResponse.json({ error: 'An unknown error occurred while processing the request' }, { status: 500 });
    }
  }
}

async function processGroqRequest(
  message: string,
  model: string = "llama-3.2-90b-text-preview"
) {
  const response = await groq.chat.completions.create({
    messages: [
      {
        role: "user",
        content: message,
      },
    ],
    model: model,
    max_tokens: 1000,
    temperature: 0.5,
    top_p: 1,
    stream: false,
  });

  return response.choices[0]?.message?.content || '';
}
