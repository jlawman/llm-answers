import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import { loadPrompt } from '@/utils/promptLoader';
import { logger } from '@/lib/braintrustlogger';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function processOpenAIRequest(
  message: string,
  model: string = "gpt-4o"
) {
  const promptTemplate = await loadPrompt('ideaPrompt');
  const prompt = promptTemplate.replace('${idea}', message);

  const response = await openai.chat.completions.create({
    model: model,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
  });

  const content = response.choices[0].message.content || '';
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
    const body = await request.json();
    const { message, model = "gpt-4o" } = body; //DANGER Leave model as GPT-4o by default

    if (!message) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    const result = await processOpenAIRequest(message, model);

    // Log asynchronously without waiting
    logger.traced(async (span) => {
      const promptTemplate = await loadPrompt('ideaPrompt');
      const fullPrompt = promptTemplate.replace('${idea}', message);
      span.log({
        input: { message, model, fullPrompt },
        output: result,
        tags: ['openai', 'request', 'response']
      });
    }, { name: 'ask_openai' }).catch(console.error);

    return NextResponse.json(result.extractedContents);
  } catch (error) {
    console.error('Error processing request:', error);
    
    // Log error asynchronously
    logger.traced(async (span) => {
      span.log({
        error: error instanceof Error ? error.message : 'Unknown error',
        tags: ['error', error instanceof Error ? 'known' : 'unknown']
      });
    }, { name: 'ask_openai_error' }).catch(console.error);

    if (error instanceof Error) {
      return NextResponse.json({ error: `Error processing request: ${error.message}` }, { status: 500 });
    } else {
      return NextResponse.json({ error: 'An unknown error occurred while processing the request' }, { status: 500 });
    }
  }
}
