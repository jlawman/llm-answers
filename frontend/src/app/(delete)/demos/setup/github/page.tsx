'use client';

import React, { useState } from 'react';
import { Container } from '@/components/ui/container';

const GitHubSetupPage: React.FC = () => {
  const [copied, setCopied] = useState(false);
  const command = 'bash -c \'cd ~/Documents/adder/"100 templates"/red-panda && ./red-panda-creator.sh\'';

  const copyToClipboard = () => {
    navigator.clipboard.writeText(command).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <Container className="py-16 sm:py-24">
      <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl mb-6">
        GitHub Setup
      </h1>
      <p className="text-lg text-gray-600 mb-8">
        To set up GitHub integration, please use the installer provided in your project.
      </p>
      <div className="bg-gray-100 p-4 rounded-md relative">
        <pre className="text-sm text-gray-800 overflow-x-auto">{command}</pre>
        <button
          onClick={copyToClipboard}
          className="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-sm hover:bg-blue-600 transition-colors"
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
    </Container>
  );
};

export default GitHubSetupPage;
