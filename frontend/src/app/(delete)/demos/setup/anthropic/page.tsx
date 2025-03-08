'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const AddAnthropicPage: React.FC = () => {
  const installCode = `npm install @anthropic-ai/sdk`;

  const usageCode = `
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function generateText(prompt: string) {
  try {
    const completion = await anthropic.completions.create({
      model: "claude-2",
      prompt: \`Human: \${prompt}\n\nAssistant:\`,
      max_tokens_to_sample: 300,
    });

    return completion.completion;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
  `;

  const envSetupCode = `
ANTHROPIC_API_KEY=your_anthropic_api_key_here
  `;

  const CodeBlock: React.FC<{ code: string; language: string }> = ({ code, language }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      toast.success('Code copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    };

    return (
      <div className="relative">
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 p-2 rounded-md bg-blue-600 hover:bg-blue-700 transition-colors"
        >
          {copied ? <Check size={18} className="text-white" /> : <Copy size={18} className="text-white" />}
        </button>
        <SyntaxHighlighter language={language} style={tomorrow}>
          {code.trim()}
        </SyntaxHighlighter>
      </div>
    );
  };

  return (
    <div className="max-w-3xl mx-auto py-8">
      <Toaster position="top-right" />
      <h1 className="text-3xl font-bold mb-6">Setting up Anthropic in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create an Anthropic account:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://www.anthropic.com" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Anthropic&apos;s website</a> and sign up</li>
            <li>Navigate to the API section and create a new API key</li>
          </ul>
        </li>

        <li>
          <strong>Install Anthropic SDK in your Next.js project:</strong>
          <CodeBlock language="bash" code={installCode} />
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2 mb-2">Add the following to your <code>.env.local</code> file:</p>
          <CodeBlock language="bash" code={envSetupCode} />
          <p className="mt-2">Replace the placeholder value with your actual Anthropic API key.</p>
        </li>

        <li>
          <strong>Implement Anthropic in your Next.js app:</strong>
          <p className="mt-2 mb-2">Here&apos;s an example of how to use Anthropic in your application:</p>
          <CodeBlock language="typescript" code={usageCode} />
        </li>

        <li>
          <strong>Use Anthropic in your components or API routes:</strong>
          <p className="mt-2">You can now use the <code>generateText</code> function in your components or API routes to interact with Anthropic&apos;s models.</p>
        </li>

        <li>
          <strong>Handle API responses and errors:</strong>
          <p className="mt-2">Implement proper error handling and response processing in your application.</p>
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Anthropic will be integrated into your Next.js application, allowing you to leverage powerful language models like Claude for various tasks such as text generation, summarization, and more.
      </p>
    </div>
  );
};

export default AddAnthropicPage;
