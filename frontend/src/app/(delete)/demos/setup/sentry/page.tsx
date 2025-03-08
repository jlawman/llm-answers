'use client';

import React from 'react';
import Lightbox from '@/components/ui/lightbox';
import Link from 'next/link';

const SetupSentryPage: React.FC = () => {

  return (
    <div className="max-w-3xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Setting up Sentry</h1>
      <ol className="list-decimal list-inside space-y-4">
        <li>Go to <a href="https://sentry.io" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">sentry.io</a></li>
        <li>
          Create a new project
          <Lightbox
            src="https://utfs.io/f/Nj1OOTmjgJCPkeOcEKPItpaLjviZ0cRo9mJqQzeyO6dAuCG8"
            alt="Creating a Sentry project"
            width={600}
            height={400}
          />
        </li>
        <li>
          Configure your new project
          <Lightbox
            src="https://utfs.io/f/Nj1OOTmjgJCPWKjKZWLx486mYNndsIxGXLErvTkbg3pROyAl"
            alt="Configuring a new Sentry project"
            width={600}
            height={400}
          />
        </li>
        <li>
          Run the command provided by Sentry, accept defaults, and then go to issues page.


          <Lightbox
            src="https://utfs.io/f/Nj1OOTmjgJCPkfdaK5PItpaLjviZ0cRo9mJqQzeyO6dAuCG8"
            alt="Running Sentry setup command"
            width={600}
            height={400}
          />
                    <p className="mt-2">
            Wizard Notes: 
            <ul className="list-disc list-inside ml-4">
              <li>Use default options (yes)</li>
              <li>Some wizard steps take a couple of minutes to run.</li>
            </ul>
            
          </p>
        </li>
        
        <li>
          Test Sentry by triggering a sample error:
          <Link href="http://localhost:3000/sentry-example-page" className="block mt-2 text-blue-500 hover:underline" target="_blank" rel="noopener noreferrer">
            Local Host Sentry Example Page (http://localhost:3000/sentry-example-page)
          </Link>
          <p className="mt-2">
            Click the button on the example page to trigger a sample error. Confirm issue appears in issues page on sentry.
          </p>
        </li>
        <li>
          Move the token from the sentry .env file to the vercel env and remove the sentry .env file.
        </li>
      </ol>
      <p className="mt-6">
        After completing these steps, Sentry will be set up to monitor your Next.js application for errors and performance issues.
      </p>
    </div>
  );
};

export default SetupSentryPage;
