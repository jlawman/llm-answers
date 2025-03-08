'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';
import { Divider } from '@/components/ui/divider';
import Lightbox from '@/components/ui/lightbox';
import { Button } from '@/components/ui/button';
const AddFathomPage: React.FC = () => {
 
  const envCode = `NEXT_PUBLIC_FATHOM_SITE_ID=YOUR_FATHOM_SITE_ID`;

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
      <h1 className="text-3xl font-bold mb-6">Setting up Fathom Analytics in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">

        <li>
          <strong>Add Fathom Site</strong>
          <p className="mt-2">Go to <a href="https://app.usefathom.com/sites" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Fathom sites page</a> and click add site</p>
          <Lightbox
            src="https://utfs.io/f/Nj1OOTmjgJCPo2dEUg1Tkg60PBGFWRtEemZHz7rxaIp1Cw4K"
            alt="Adding a Fathom site"
            width={600}
            height={400}
          />
          <p className="mt-2">Name and save site:</p>
          <Lightbox
            src="https://utfs.io/f/Nj1OOTmjgJCPzlGmu0TNijfEtx2bpCdFGXkmH70g9rMcWL5Z"
            alt="Naming and saving Fathom site"
            width={600}
            height={400}
          />
          
        </li>

    

        <li>
          <strong>Add your Fathom site ID to .env.local:</strong>
          <Lightbox
            src="https://utfs.io/f/Nj1OOTmjgJCPGwZkmwdavbqgwdDipKN3S6MB1yC8Z9X4zPfR"
            alt="Copying Fathom site ID"
            width={600}
            height={400}
          />
          <p className="mt-2">Update vercel env:</p>
          <CodeBlock language="bash" code={envCode} />
          
        </li>

        <li>
          <strong>(Optional) Verify installation:</strong>
          <p className="mt-2">Check <a href="https://app.usefathom.com/sites" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Fathom sites page</a> and go to your site to ensure data is being collected.</p>
        </li>
        <li>
          <strong>(Optional) Add logging of events:</strong>
          <p className="mt-2 mb-2">Go to demo for logging events.</p>
          <Button
            onClick={() => window.location.href = '/monitoring/log-fathom-events'}
            className="mt-2"
          >
            Fathom Event Logging Demo
          </Button>
        </li>
        <Divider />
        <h2>If you aren&apos;t using the template, also go to <a href="https://vercel.com/guides/deploying-nextjs-using-fathom-analytics-with-vercel" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Vercel&apos;s Fathom install guide</a></h2>
        
      </ol>
      <p className="mt-6">
        After completing these steps, Fathom Analytics will be integrated into your Next.js application, providing privacy-friendly analytics.
      </p>
    </div>
  );
};

export default AddFathomPage;
