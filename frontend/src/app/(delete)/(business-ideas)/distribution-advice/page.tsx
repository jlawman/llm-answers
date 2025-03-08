'use client';

import { useState } from 'react';
import { Container } from '@/components/ui/container';
import ReactMarkdown from 'react-markdown';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Tabs } from "@/components/ui/tabs";

export default function DistributionAdvice() {
  const [url, setUrl] = useState('');
  const [description, setDescription] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('URL');

  const isValidUrl = (url: string) => {
    let formattedUrl = url;
    
    // If URL doesn't start with http:// or https://, prepend https://
    if (!formattedUrl.match(/^https?:\/\//i)) {
      formattedUrl = 'https://' + formattedUrl;
    }

    // If URL doesn't have a dot after the protocol, add www.
    if (!formattedUrl.match(/^https?:\/\/.*\./i)) {
      formattedUrl = formattedUrl.replace(/^(https?:\/\/)/, '$1www.');
    }

    try {
      new URL(formattedUrl);
      return formattedUrl; // Return the formatted URL instead of just true
    } catch (e) {
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      if (activeTab === 'URL') {
        const validUrl = isValidUrl(url);
        if (!validUrl) {
          setError('Please enter a valid URL');
          setLoading(false);
          return;
        }

        const parseRes = await fetch('/api/parse-page', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ url: validUrl }), // Use the formatted URL
        });

        const parseData = await parseRes.json();
        
        if (!parseRes.ok) {
          throw new Error(parseData.error || 'Failed to parse URL');
        }

        const groqRes = await fetch('/api/ask-groq', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: parseData.markdown }),
        });

        const groqData = await groqRes.json();
        
        if (!groqRes.ok) {
          throw new Error(groqData.error || 'Failed to get response');
        }

        setResponse(groqData.result);
      } else {
        if (!description.trim()) {
          setError('Please enter a description');
          setLoading(false);
          return;
        }

        const groqRes = await fetch('/api/ask-groq', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ message: description }),
        });

        const groqData = await groqRes.json();
        
        if (!groqRes.ok) {
          throw new Error(groqData.error || 'Failed to get response');
        }

        setResponse(groqData.result);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="py-8">
      <h1 className="text-3xl font-bold mb-8">Distribution Advice</h1>
      
      <div className="mb-12 space-y-4">
        <p className="text-gray-600 text-sm mb-6">
          Distribution advice to maximize your reach and engagement. Either enter a URL to analyze existing content
          or describe your SaaS product directly. The advice assumes you are able to build AI tooling to help you distribute your product.
        </p>

        <Tabs
          tabs={[
            { name: 'URL', current: activeTab === 'URL' },
            { name: 'Description', current: activeTab === 'Description' }
          ]}
          onChange={(tab) => setActiveTab(tab.name)}
        />

        <form onSubmit={handleSubmit} className="space-y-4">
          {activeTab === 'URL' ? (
            <div>
              <input
                type="text"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="Enter URL to analyze..."
                className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                required
              />
            </div>
          ) : (
            <div>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your SaaS product, target market, and distribution goals..."
                className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 min-h-[100px]"
                required
              />
            </div>
          )}
          
          <Button
            type="submit"
            disabled={loading || (!url.trim() && !description.trim())}
            className="w-full sm:w-auto"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 inline animate-spin" />
                Analyzing...
              </>
            ) : (
              'Get Distribution Advice'
            )}
          </Button>
        </form>
      </div>

      {error && (
        <div className="mb-8 p-4 bg-red-50 text-red-500 border border-red-200 rounded-lg">
          {error}
        </div>
      )}

      {response && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold mb-4">Distribution Strategy:</h2>
          <ReactMarkdown className="prose prose-sm max-w-none">
            {response}
          </ReactMarkdown>
        </div>
      )}
    </Container>
  );
}
