import { NextRequest, NextResponse } from 'next/server';
import { Anthropic } from '@anthropic-ai/sdk';
import { loadPrompt } from '@/utils/promptLoader';
import { logger } from '@/lib/braintrustlogger';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function processAnthropicRequest(
  message: string,
  promptTemplateName: string,
  model: string = 'claude-3-5-sonnet-latest'
) {
  if (!promptTemplateName) {
    throw new Error('No prompt template specified');
  }

  //console.log('Received prompt template name:', promptTemplateName);
  //console.log('Received message:', message);
  const promptTemplate = await loadPrompt(promptTemplateName);
  const prompt = promptTemplate.replace('${userInput}', message);

  const response = await anthropic.messages.create({
    model: model,
    max_tokens: 1000,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: prompt,
          },
        ],
      },
    ],
  });

  const content = (response.content[0] as { type: string; text: string }).text;
  
  const xmlTags = content.match(/<(\w+)>([\s\S]*?)<\/\1>/g) || [];
  const xmlContents: { [key: string]: string } = {};
  const logoSuggestions: Array<{ name: string; description: string }> = [];

  xmlTags.forEach(tag => {
    const match = tag.match(/<(\w+)>([\s\S]*?)<\/\1>/);
    if (match) {
      const [, tagName, tagContent] = match;
      xmlContents[tagName] = tagContent.trim();
      
      if (tagName.startsWith('logo_name_')) {
        const number = tagName.split('_')[2];
        const descriptionKey = `logo_description_${number}`;
        if (xmlContents[descriptionKey]) {
          logoSuggestions.push({
            name: xmlContents[tagName],
            description: xmlContents[descriptionKey]
          });
        }
      }
    }
  });

  return {
    llmResponse: content,
    extractedContents: xmlContents,
    logoSuggestions
  };
}

export async function POST(request: NextRequest): Promise<NextResponse> {
  try {
    const { message, model, promptTemplateName } = await request.json();

    if (!message) {
      return NextResponse.json({ error: 'No message provided' }, { status: 400 });
    }

    if (!promptTemplateName) {
      return NextResponse.json({ error: 'No prompt template specified' }, { status: 400 });
    }

    const result = await processAnthropicRequest(message, promptTemplateName, model);

    //console.log('API Response:', result); // Add this line

    // Log asynchronously without waiting
    logger.traced(async (span) => {
      const promptTemplate = await loadPrompt(promptTemplateName);
      const fullPrompt = promptTemplate.replace('${userInput}', message);
      span.log({
        input: { message, model, promptTemplateName, fullPrompt },
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
