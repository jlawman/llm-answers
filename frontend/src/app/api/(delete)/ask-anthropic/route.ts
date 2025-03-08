import { NextRequest, NextResponse } from 'next/server';
import { Anthropic } from '@anthropic-ai/sdk';
import { logger } from '@/lib/braintrustlogger';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function processAnthropicRequest(
  message: string,
  model: string = 'claude-3-5-sonnet-latest'
) {
  // Log the incoming request

  const response = await anthropic.messages.create({
    model: model,
    max_tokens: 1000,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: message,
          },
        ],
      },
    ],
  });

  const content = (response.content[0] as { type: string; text: string }).text;
  
  const xmlTags = content.match(/<(\w+)>([\s\S]*?)<\/\1>/g) || [];
  const xmlContents: { [key: string]: string } = {};

  xmlTags.forEach(tag => {
    const match = tag.match(/<(\w+)>([\s\S]*?)<\/\1>/);
    if (match) {
      const [, tagName, tagContent] = match;
      xmlContents[tagName] = tagContent.trim();
    }
  });


  return {
    llmResponse: content,
    extractedContents: xmlContents
  };
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const { message, model } = await request.json();

    if (!message) {
      return NextResponse.json({ error: 'No message provided' }, { status: 400 });
    }

    const result = await processAnthropicRequest(message, model);

    // Log asynchronously without waiting
    logger.traced(async (span) => {
      span.log({
        input: { message, model },
        output: result,
        tags: ['anthropic', 'request', 'response']
      });
    }, { name: 'ask_anthropic' }).catch(console.error);

    return NextResponse.json(result.extractedContents);
  } catch (error) {
    console.error('Error processing request:', error);
    
    // Log error asynchronously
    logger.traced(async (span) => {
      span.log({
        error: error instanceof Error ? error.message : 'Unknown error',
        tags: ['error', error instanceof Error ? 'known' : 'unknown']
      });
    }, { name: 'ask_anthropic_error' }).catch(console.error);

    if (error instanceof Error) {
      return NextResponse.json({ error: `Error processing request: ${error.message}` }, { status: 500 });
    } else {
      return NextResponse.json({ error: 'An unknown error occurred while processing the request' }, { status: 500 });
    }
  }
}
