'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const AddBraintrustPage: React.FC = () => {
  const installCode = `npm install @braintrust/sdk`;

  const usageCode = `
import { Braintrust } from '@braintrust/sdk';

const braintrust = new Braintrust({
  apiKey: process.env.BRAINTRUST_API_KEY,
});

// Example usage
async function evaluateModel() {
  const result = await braintrust.evaluate({
    model: 'gpt-3.5-turbo',
    prompt: 'Translate the following English text to French: "Hello, how are you?"',
    response: 'Bonjour, comment allez-vous ?',
  });

  console.log(result);
}
  `;

  const envSetupCode = `
BRAINTRUST_API_KEY=your_braintrust_api_key_here
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
      <h1 className="text-3xl font-bold mb-6">Setting up Braintrust in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create a Braintrust account:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://www.braintrustdata.com/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Braintrust&apos;s website</a> and sign up</li>
            <li>Obtain your API key from the dashboard</li>
          </ul>
        </li>

        <li>
          <strong>Install Braintrust SDK in your Next.js project:</strong>
          <CodeBlock language="bash" code={installCode} />
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2 mb-2">Add the following to your <code>.env.local</code> file:</p>
          <CodeBlock language="bash" code={envSetupCode} />
          <p className="mt-2">Replace <code>your_braintrust_api_key_here</code> with your actual Braintrust API key.</p>
        </li>

        <li>
          <strong>Implement Braintrust in your Next.js app:</strong>
          <p className="mt-2 mb-2">Here&apos;s an example of how to use Braintrust in your application:</p>
          <CodeBlock language="javascript" code={usageCode} />
        </li>

        <li>
          <strong>Use Braintrust for model evaluation:</strong>
          <p className="mt-2">Integrate Braintrust into your AI workflows to evaluate and monitor your models.</p>
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Braintrust will be integrated into your Next.js application, allowing you to evaluate and monitor your AI models.
      </p>
    </div>
  );
};

export default AddBraintrustPage;
