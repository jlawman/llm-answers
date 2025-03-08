'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';

const AddStripePage: React.FC = () => {
  const companyNumberExample = '123';
  const companyNameExample = 'My Company Name';

  const clientSideCode = `
import { loadStripe } from '@stripe/stripe-js';
import { Elements } from '@stripe/react-stripe-js';

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!);

export default function CheckoutPage() {
  return (
    <Elements stripe={stripePromise}>
      {/* Your checkout form components go here */}
    </Elements>
  );
}
  `;

  const envSetupCode = `
STRIPE_SECRET_KEY=your_stripe_secret_key_here
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key_here
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
      <h1 className="text-3xl font-bold mb-6">Setting up Stripe in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create a Stripe account:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://dashboard.stripe.com/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Stripe Dashboard</a> and create a new account</li>
            <li>Click activate payments button</li>
            <li>Go to <a href="https://find-and-update.company-information.service.gov.uk/company/13295099" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Companies House</a> and for your company and use the company number to create a Stripe account</li>
          </ul>
        </li>

        <li>
          <strong>Company Information:</strong>
          <p className="mt-2 mb-2">Use the following information to create your Stripe account:</p>
          <div className="space-y-4">
            <p className="mt-4">Company Number:</p>
            <CodeBlock language="plaintext" code={`${companyNumberExample}`} />
            <p>Company Name:</p>
            <CodeBlock language="plaintext" code={`${companyNameExample}`} />
          </div>
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2 mb-2">Add the following to your <code>.env.local</code> file:</p>
          <CodeBlock language="bash" code={envSetupCode} />
          <p className="mt-2">Replace the placeholder values with your actual Stripe API keys.</p>
        </li>

        <li>
          <strong>Implement Stripe on the client-side:</strong>
          <p className="mt-2 mb-2">Set up Stripe Elements in your checkout page:</p>
          <CodeBlock language="typescript" code={clientSideCode} />
        </li>

        <li>
          <strong>Handle payments:</strong>
          <p className="mt-2">Implement the necessary logic to create payment intents, handle successful payments, and manage webhooks for asynchronous events.</p>
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Stripe will be integrated into your Next.js application, allowing you to process payments securely. Remember to thoroughly test your implementation and follow Stripe&apos;s best practices for security and compliance.
      </p>
    </div>
  );
};

export default AddStripePage;
