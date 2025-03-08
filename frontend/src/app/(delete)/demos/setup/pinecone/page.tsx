'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const AddPineconePage: React.FC = () => {
  const installCode = `npm install @pinecone-database/pinecone`;

  const usageCode = `
import { PineconeClient } from '@pinecone-database/pinecone';

const pinecone = new PineconeClient();

await pinecone.init({
  environment: process.env.PINECONE_ENVIRONMENT,
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

// Example: Upserting vectors
const upsertRequest = {
  vectors: [
    {
      id: 'vec1',
      values: [0.1, 0.2, 0.3, 0.4],
      metadata: { text: 'Example text' }
    }
  ],
  namespace: 'example-namespace'
};

await index.upsert({ upsertRequest });

// Example: Querying vectors
const queryRequest = {
  vector: [0.1, 0.2, 0.3, 0.4],
  topK: 10,
  includeValues: true,
  includeMetadata: true,
  namespace: 'example-namespace'
};

const queryResponse = await index.query({ queryRequest });
console.log(queryResponse);
  `;

  const envSetupCode = `
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here
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
      <h1 className="text-3xl font-bold mb-6">Setting up Pinecone in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create a Pinecone account:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://www.pinecone.io/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Pinecone&apos;s website</a> and sign up</li>
            <li>Create a new project and index in the Pinecone dashboard</li>
          </ul>
        </li>

        <li>
          <strong>Install Pinecone SDK in your Next.js project:</strong>
          <CodeBlock language="bash" code={installCode} />
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2 mb-2">Add the following to your <code>.env.local</code> file:</p>
          <CodeBlock language="bash" code={envSetupCode} />
          <p className="mt-2">Replace the placeholder values with your actual Pinecone credentials.</p>
        </li>

        <li>
          <strong>Implement Pinecone in your Next.js app:</strong>
          <p className="mt-2 mb-2">Here&apos;s an example of how to use Pinecone in your application:</p>
          <CodeBlock language="javascript" code={usageCode} />
        </li>

        <li>
          <strong>Use Pinecone for vector storage and similarity search:</strong>
          <p className="mt-2">Integrate Pinecone into your AI workflows for efficient vector storage and retrieval.</p>
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Pinecone will be integrated into your Next.js application, allowing you to store and query vector embeddings for AI-powered features.
      </p>
    </div>
  );
};

export default AddPineconePage;
