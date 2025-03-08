'use client';

import { useState } from 'react';
import { Container } from '@/components/ui/container';
import ReactMarkdown from 'react-markdown';
import { Loader2 } from 'lucide-react';

export default function WebScrape() {
  const [url, setUrl] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const res = await fetch('/api/parse-page', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Failed to parse URL');
      }

      setResponse(data.markdown);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="py-8">
      <h1 className="text-3xl font-bold mb-8">Web Page Parser</h1>
      
      <div className="mb-12 space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter URL to parse"
              className="w-full p-2 border rounded-md"
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Parsing...
              </span>
            ) : (
              'Parse URL'
            )}
          </button>
        </form>

        {error && (
          <div className="p-4 text-red-500 bg-red-50 rounded-md">
            {error}
          </div>
        )}

        {response && (
          <div className="mt-8">
            <h2 className="text-xl font-semibold mb-4">Parsed Content</h2>
            <div className="prose max-w-none">
              <ReactMarkdown>{response}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </Container>
  );
}
