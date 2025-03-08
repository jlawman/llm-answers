'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const AddGroqPage: React.FC = () => {
  const installCode = `npm install groq-sdk`;

  const usageCode = `
import Groq from 'groq-sdk';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

async function generateText(prompt: string) {
  try {
    const chatCompletion = await groq.chat.completions.create({
      messages: [{ role: 'user', content: prompt }],
      model: 'mixtral-8x7b-32768',
    });

    return chatCompletion.choices[0]?.message?.content;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
  `;

  const envSetupCode = `
GROQ_API_KEY=your_groq_api_key_here
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
      <h1 className="text-3xl font-bold mb-6">Setting up Groq in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create a Groq account:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://console.groq.com" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Groq&apos;s console</a> and sign up</li>
            <li>Navigate to the API section and create a new API key</li>
          </ul>
        </li>

        <li>
          <strong>Install Groq SDK in your Next.js project:</strong>
          <CodeBlock language="bash" code={installCode} />
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2 mb-2">Add the following to your <code>.env.local</code> file:</p>
          <CodeBlock language="bash" code={envSetupCode} />
          <p className="mt-2">Replace the placeholder value with your actual Groq API key.</p>
        </li>

        <li>
          <strong>Implement Groq in your Next.js app:</strong>
          <p className="mt-2 mb-2">Here&apos;s an example of how to use Groq in your application:</p>
          <CodeBlock language="typescript" code={usageCode} />
        </li>

        <li>
          <strong>Use Groq in your components or API routes:</strong>
          <p className="mt-2">You can now use the <code>generateText</code> function in your components or API routes to interact with Groq&apos;s models.</p>
        </li>

        <li>
          <strong>Handle API responses and errors:</strong>
          <p className="mt-2">Implement proper error handling and response processing in your application.</p>
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Groq will be integrated into your Next.js application, allowing you to leverage their high-performance language models for various tasks such as text generation, summarization, and more.
      </p>
    </div>
  );
};

export default AddGroqPage;
