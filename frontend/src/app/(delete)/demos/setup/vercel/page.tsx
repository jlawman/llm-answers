'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const AddVercelPage: React.FC = () => {
  const vercelJsonCode = `
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ]
}
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
      <h1 className="text-3xl font-bold mb-6">Setting up Vercel Deployment for Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create a Vercel account:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://vercel.com/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Vercel&apos;s website</a> and sign up</li>
            <li>Connect your GitHub, GitLab, or Bitbucket account</li>
          </ul>
        </li>

        <li>
          <strong>Prepare your Next.js project:</strong>
          <p className="mt-2">Ensure your project is pushed to a Git repository.</p>
        </li>

        <li>
          <strong>Configure Vercel deployment:</strong>
          <p className="mt-2 mb-2">Create a <code>vercel.json</code> file in your project root (optional):</p>
          <CodeBlock language="json" code={vercelJsonCode} />
        </li>

        <li>
          <strong>Deploy your project:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to your Vercel dashboard and click &quot;Import Project&quot;</li>
            <li>Select your Git repository</li>
            <li>Configure your project settings (build command, output directory, etc.)</li>
            <li>Click &quot;Deploy&quot;</li>
          </ul>
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2">Add any necessary environment variables in your Vercel project settings.</p>
        </li>

        <li>
          <strong>Configure custom domain (optional):</strong>
          <p className="mt-2">Add and configure a custom domain in your Vercel project settings.</p>
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, your Next.js application will be deployed on Vercel, providing a fast and reliable hosting solution with automatic deployments on every push to your repository.
      </p>
    </div>
  );
};

export default AddVercelPage;
