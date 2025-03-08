'use client';

import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { toast, Toaster } from 'sonner';
import Lightbox from '@/components/ui/lightbox';

const AddClerkPage: React.FC = () => {
  const layoutCode = `import { ClerkProvider, SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs'
import './globals.css'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <ClerkProvider>
      <html lang="en">
        <body>
          <header>
            <SignedOut>
              <SignInButton />
            </SignedOut>
            <SignedIn>
              <UserButton />
            </SignedIn>
          </header>
          <main>{children}</main>
        </body>
      </html>
    </ClerkProvider>
  )
}
  

Add Clerk to the layout. See example above for up to date way to do this.
`;




  const envSetupCode = `
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_publishable_key_here
CLERK_SECRET_KEY=your_secret_key_here
NEXT_PUBLIC_CLERK_SIGN_IN_URL=/sign-in
NEXT_PUBLIC_CLERK_SIGN_UP_URL=/sign-up
NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL=/
NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL=/
  `;

  const middlewareCode = `import { clerkMiddleware } from '@clerk/nextjs/server'

export default clerkMiddleware()

export const config = {
  matcher: [
    // Skip Next.js internals and all static files, unless found in search params
    '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
    // Always run for API routes
    '/(api|trpc)(.*)',
  ],
}
  
Add this clerk funcitonality to the existing middleware file. All existing functionality should be preserved exactly as it was.
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

  const ZoomableImage: React.FC<{ src: string; alt: string; width: number; height: number }> = ({ src, alt, width, height }) => {
    return (
      <Lightbox
        src={src}
        alt={alt}
        width={width}
        height={height}
      />
    );
  };

  return (
    <div className="max-w-3xl mx-auto py-8">
      <Toaster position="top-right" />
      <h1 className="text-3xl font-bold mb-6">Setting up Clerk in Next.js</h1>
      <ol className="list-decimal list-inside space-y-6">
        <li>
          <strong>Create a Clerk account and project:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Go to <a href="https://dashboard.clerk.com/" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">Clerk&apos;s dashboard</a> and sign up or log in</li>
            <li>Create a new project and configure it</li>
            <li>Follow the directions in their quickstart guide <a href="https://clerk.com/docs/quickstarts/nextjs" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">here</a></li>
          </ul>
        </li>

        <li>
          <strong>Install Clerk in your Next.js project:</strong>
          <CodeBlock language="bash" code="npm install @clerk/nextjs" />
        </li>

        <li>
          <strong>Set up environment variables:</strong>
          <p className="mt-2 mb-2">Create a <code>.env.local</code> file in your project root and add the following:</p>
          <CodeBlock language="bash" code={envSetupCode} />
          <p className="mt-2">Replace <code>your_publishable_key_here</code> and <code>your_secret_key_here</code> with your actual Clerk keys.</p>
        </li>

        <li>
          <strong>Add Clerk Provider to your layout:</strong>
          <p className="mt-2 mb-2">Update your <code>app/layout.tsx</code> file:</p>
          <CodeBlock language="tsx" code={layoutCode} />
        </li>

        <li>
          <strong>Set up middleware for protected routes:</strong>
          <p className="mt-2 mb-2">Create or update <code>middleware.ts</code> in your project root:</p>
          <CodeBlock language="typescript" code={middlewareCode} />
        </li>

        <li>
          <strong>Add authentication to Vercel:</strong>
          <p className="mt-2">Add the new live keys to your Vercel project in the Environment Variables section.</p>
        </li>

        <li>
          <strong>Create a production instance:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Click &quot;Development&quot; at the top of the clerk dashboard and create a production instance</li>
            <li>Clone the development instance</li>
            <li>Select the primary domain option</li>
          </ul>
        </li>

        <li>
          <strong>Configure DNS settings:</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Configure CNAME for frontend API</li>
            <li>Add CNAMEs for account portal and email</li>
            <li>Wait for SSL setup to complete</li>
          </ul>
        </li>

        <li>
          <strong>Optional: Customize Clerk&apos;s appearance</strong>
          <ul className="list-disc list-inside ml-6 mt-2 space-y-2">
            <li>Add logo and favicon in Clerk&apos;s Application settings</li>
            <li>Update logo color in account portal link (or use signout button)</li>
          </ul>
        </li>

        <li>
          <strong>Optional: Add additional authentication methods</strong>
          <p className="mt-2">For example, to add Google authentication, follow Clerk&apos;s guide:</p>
          <ZoomableImage
            src="https://utfs.io/f/Nj1OOTmjgJCPA0chbnqXVmizEN9FMl3djKLrYgsW4QcAhTG1"
            alt="Clerk Google Authentication Guide"
            width={600}
            height={400}
          />
          <ZoomableImage
            src="https://utfs.io/f/Nj1OOTmjgJCPjZPh19TIJEYmCWB4TtZQ0KahDqPXM7kLVNOg"
            alt="Additional Clerk Authentication Options"
            width={600}
            height={400}
          />
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Clerk will be fully integrated into your Next.js application, handling authentication and user management.
      </p>
    </div>
  );
};

export default AddClerkPage;
