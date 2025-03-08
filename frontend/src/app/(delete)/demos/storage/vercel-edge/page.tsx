'use client';

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const VercelEdgePage: React.FC = () => {
  const [key, setKey] = useState('');
  const [value, setValue] = useState('');
  const [exists, setExists] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [allKeys, setAllKeys] = useState<string>('');
  const [batchKeys, setBatchKeys] = useState<string>('');
  const [batchValues, setBatchValues] = useState<string>('');

  const handleRead = async () => {
    try {
      const response = await fetch(`/api/edge-config/read?key=${encodeURIComponent(key)}`);
      if (!response.ok) throw new Error('Failed to read value');
      const result = await response.json();
      setValue(JSON.stringify(result, null, 2));
      setError(null);
    } catch (err) {
      setError('Failed to read value');
      setValue('');
    }
  };

  const handleCheckExists = async () => {
    try {
      const response = await fetch(`/api/edge-config/exists?key=${encodeURIComponent(key)}`);
      if (!response.ok) throw new Error('Failed to check if key exists');
      const result = await response.json();
      setExists(result.exists);
      setError(null);
    } catch (err) {
      setError('Failed to check if key exists');
      setExists(null);
    }
  };

  const handleGetAll = async () => {
    try {
      const response = await fetch('/api/edge-config/get-all');
      if (!response.ok) throw new Error('Failed to get all keys and values');
      const result = await response.json();
      setAllKeys(JSON.stringify(result, null, 2));
      setError(null);
    } catch (err) {
      setError('Failed to get all keys and values');
      setAllKeys('');
    }
  };

  const handleReadBatch = async () => {
    try {
      const keys = batchKeys.split(',').map(k => k.trim());
      const response = await fetch('/api/edge-config/read-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keys }),
      });
      if (!response.ok) throw new Error('Failed to read batch values');
      const result = await response.json();
      setBatchValues(JSON.stringify(result, null, 2));
      setError(null);
    } catch (err) {
      setError('Failed to read batch values');
      setBatchValues('');
    }
  };

  useEffect(() => {
    setExists(null);
    setValue('');
    setError(null);
  }, [key]);

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-3xl font-extrabold text-gray-900 mb-8">Vercel Edge Config Demo</h1>
        
        <div className="bg-white shadow sm:rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h2 className="text-lg leading-6 font-medium text-gray-900 mb-4">Read from Edge Config</h2>
            <div className="mt-2 max-w-xl text-sm text-gray-500">
              <p>Enter a key to read its value from Vercel Edge Config.</p>
            </div>
            <div className="mt-5 sm:flex sm:items-center">
              <div className="max-w-xs w-full">
                <Input
                  type="text"
                  placeholder="Enter key"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                />
              </div>
              <div className="mt-3 sm:mt-0 sm:ml-4 sm:flex-shrink-0">
                <Button onClick={handleRead}>Read Value</Button>
              </div>
              <div className="mt-3 sm:mt-0 sm:ml-4 sm:flex-shrink-0">
                <Button variant="secondary" onClick={handleCheckExists}>Check Exists</Button>
              </div>
            </div>
            {error && (
              <p className="mt-2 text-sm text-red-600">{error}</p>
            )}
            {value && (
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900">Value:</h3>
                <pre className="mt-2 text-sm text-gray-500 bg-gray-50 p-4 rounded-md overflow-auto">{value}</pre>
              </div>
            )}
            {exists !== null && (
              <p className="mt-2 text-sm text-gray-600">
                Key {exists ? 'exists' : 'does not exist'} in Edge Config.
              </p>
            )}
          </div>
        </div>

        <div className="bg-white shadow sm:rounded-lg mt-8">
          <div className="px-4 py-5 sm:p-6">
            <h2 className="text-lg leading-6 font-medium text-gray-900 mb-4">Get All Keys and Values</h2>
            <div className="mt-2 max-w-xl text-sm text-gray-500">
              <p>Retrieve all keys and values from Vercel Edge Config.</p>
            </div>
            <div className="mt-5">
              <Button onClick={handleGetAll}>Get All</Button>
            </div>
            {allKeys && (
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900">All Keys and Values:</h3>
                <pre className="mt-2 text-sm text-gray-500 bg-gray-50 p-4 rounded-md overflow-auto">{allKeys}</pre>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white shadow sm:rounded-lg mt-8">
          <div className="px-4 py-5 sm:p-6">
            <h2 className="text-lg leading-6 font-medium text-gray-900 mb-4">Read Multiple Keys</h2>
            <div className="mt-2 max-w-xl text-sm text-gray-500">
              <p>Enter comma-separated keys to read their values in batch.</p>
            </div>
            <div className="mt-5 sm:flex sm:items-center">
              <div className="max-w-xs w-full">
                <Input
                  type="text"
                  placeholder="Enter keys (e.g., keyA, keyB)"
                  value={batchKeys}
                  onChange={(e) => setBatchKeys(e.target.value)}
                />
              </div>
              <div className="mt-3 sm:mt-0 sm:ml-4 sm:flex-shrink-0">
                <Button onClick={handleReadBatch}>Read Batch</Button>
              </div>
            </div>
            {batchValues && (
              <div className="mt-4">
                <h3 className="text-lg font-medium text-gray-900">Batch Values:</h3>
                <pre className="mt-2 text-sm text-gray-500 bg-gray-50 p-4 rounded-md overflow-auto">{batchValues}</pre>
              </div>
            )}
          </div>
        </div>

        {error && (
          <p className="mt-4 text-sm text-red-600">{error}</p>
        )}
      </div>
    </div>
  );
};

export default VercelEdgePage;
