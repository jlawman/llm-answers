'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Toaster, toast } from 'sonner';
import { Loader2 } from 'lucide-react';

type SearchResult = {
  text: string;
  score: number;
};

const PineconeManager = () => {
  const [indexName, setIndexName] = useState('');
  const [indexes, setIndexes] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState('');
  const [customData, setCustomData] = useState('');
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [namespace, setNamespace] = useState('');
  const [loadingActions, setLoadingActions] = useState<Record<string, boolean>>({});
  const [deleteNamespace, setDeleteNamespace] = useState('');

  const fetchIndexes = useCallback(async () => {
    setActionLoading('fetchIndexes', true);
    try {
      const response = await fetch(`/api/list-pinecone-indexes?t=${Date.now()}`);
      const data = await response.json();
      if (data.success) {
        setIndexes(data.indexes);
      } else {
        toast.error('Failed to fetch indexes');
      }
    } catch (error) {
      toast.error('Error fetching indexes');
    } finally {
      setActionLoading('fetchIndexes', false);
    }
  }, []);  // Empty dependency array since it doesn't use any props or state

  useEffect(() => {
    fetchIndexes();
  }, [fetchIndexes]);  // Now we can safely add fetchIndexes as a dependency

  const setActionLoading = (action: string, isLoading: boolean) => {
    setLoadingActions(prev => ({ ...prev, [action]: isLoading }));
  };

  const createIndex = async () => {
    if (!indexName) {
      toast.error('Please enter an index name');
      return;
    }
    setActionLoading('createIndex', true);
    try {
      const response = await fetch('/api/create-pinecone-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indexName, dimension: 1024, metric: 'cosine' }),
      });
      const data = await response.json();
      if (data.success) {
        toast.success('Index created successfully');
        fetchIndexes();
        setIndexName('');
      } else {
        toast.error(data.error || 'Failed to create index');
      }
    } catch (error) {
      toast.error('Error creating index');
    } finally {
      setActionLoading('createIndex', false);
    }
  };

  const addItems = async () => {
    if (!selectedIndex) {
      toast.error('Please select an index');
      return;
    }
    if (!customData.trim()) {
      toast.error('Please enter some data');
      return;
    }
    setActionLoading('addItems', true);
    try {
      const items = customData.split('\n').map((text, index) => ({
        id: `${Date.now()}-${index}`,
        text: text.trim(),
      }));
      const response = await fetch('/api/upload-to-pinecone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indexName: selectedIndex, data: items }),
      });
      const data = await response.json();
      if (data.success) {
        toast.success('Items added successfully');
        setCustomData('');
      } else {
        toast.error(data.error || 'Failed to add items');
      }
    } catch (error) {
      toast.error('Error adding items');
    } finally {
      setActionLoading('addItems', false);
    }
  };

  const searchItems = async () => {
    if (!selectedIndex) {
      toast.error('Please select an index');
      return;
    }
    if (!query.trim()) {
      toast.error('Please enter a search query');
      return;
    }
    setActionLoading('searchItems', true);
    try {
      const response = await fetch('/api/search-pinecone-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indexName: selectedIndex, query, namespace }),
      });
      const data = await response.json();
      if (data.success) {
        setSearchResults(data.results);
        toast.success('Search completed');
      } else {
        toast.error(data.error || 'Search failed');
      }
    } catch (error) {
      toast.error('Error searching items');
    } finally {
      setActionLoading('searchItems', false);
    }
  };

  const deleteAllItems = async () => {
    if (!selectedIndex) {
      toast.error('Please select an index');
      return;
    }
    if (!confirm('Are you sure you want to delete all items from this index?')) {
      return;
    }
    setActionLoading('deleteAllItems', true);
    try {
      const response = await fetch('/api/delete-all-from-pinecone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indexName: selectedIndex }),
      });
      const data = await response.json();
      if (data.success) {
        toast.success('All items deleted successfully');
        setSearchResults([]);
      } else {
        toast.error(data.error || 'Failed to delete items');
      }
    } catch (error) {
      toast.error('Error deleting items');
    } finally {
      setActionLoading('deleteAllItems', false);
    }
  };

  const deleteIndex = async () => {
    if (!selectedIndex) {
      toast.error('Please select an index to delete');
      return;
    }
    if (!confirm(`Are you sure you want to delete the index "${selectedIndex}"? This action cannot be undone.`)) {
      return;
    }
    setActionLoading('deleteIndex', true);
    try {
      const response = await fetch('/api/delete-pinecone-index', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indexName: selectedIndex }),
      });
      const data = await response.json();
      if (data.success) {
        toast.success('Index deleted successfully');
        fetchIndexes();
        setSelectedIndex('');
        setSearchResults([]);
      } else {
        toast.error(data.error || 'Failed to delete index');
      }
    } catch (error) {
      toast.error('Error deleting index');
    } finally {
      setActionLoading('deleteIndex', false);
    }
  };

  const deleteNamespaceItems = async () => {
    if (!selectedIndex) {
      toast.error('Please select an index');
      return;
    }
    if (!deleteNamespace.trim()) {
      toast.error('Please enter a namespace');
      return;
    }
    if (!confirm(`Are you sure you want to delete all items from the namespace "${deleteNamespace}" in this index?`)) {
      return;
    }
    setActionLoading('deleteNamespaceItems', true);
    try {
      const response = await fetch('/api/delete-pinecone-namespace-records', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ indexName: selectedIndex, namespace: deleteNamespace }),
      });
      const data = await response.json();
      if (data.success) {
        toast.success(data.message);
        setSearchResults([]);
        setDeleteNamespace('');
      } else {
        toast.error(data.error || 'Failed to delete items from namespace');
      }
    } catch (error) {
      toast.error('Error deleting items from namespace');
    } finally {
      setActionLoading('deleteNamespaceItems', false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <Toaster position="top-right" />
      <h1 className="text-3xl font-bold mb-6">Pinecone Vector Database Manager</h1>
      
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Create New Index</h2>
        <div className="flex">
          <input
            type="text"
            value={indexName}
            onChange={(e) => setIndexName(e.target.value)}
            placeholder="Enter new index name"
            className="border p-2 mr-2 flex-grow"
          />
          <button
            onClick={createIndex}
            disabled={loadingActions['createIndex']}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors flex items-center"
          >
            {loadingActions['createIndex'] ? (
              <><Loader2 className="animate-spin mr-2" size={18} /> Creating...</>
            ) : (
              'Create Index'
            )}
          </button>
        </div>
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Select Index</h2>
        <div className="flex items-center">
          <select
            value={selectedIndex}
            onChange={(e) => setSelectedIndex(e.target.value)}
            className="border p-2 flex-grow mr-2"
            disabled={loadingActions['fetchIndexes']}
          >
            <option value="">
              {loadingActions['fetchIndexes'] ? 'Loading indexes...' : 'Select an index'}
            </option>
            {indexes.map((index) => (
              <option key={index} value={index}>
                {index}
              </option>
            ))}
          </select>
          <button
            onClick={deleteIndex}
            disabled={loadingActions['deleteIndex']}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors flex items-center"
          >
            {loadingActions['deleteIndex'] ? (
              <><Loader2 className="animate-spin mr-2" size={18} /> Deleting...</>
            ) : (
              'Delete Index'
            )}
          </button>
        </div>
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Add Items</h2>
        <textarea
          value={customData}
          onChange={(e) => setCustomData(e.target.value)}
          placeholder="Enter items (one per line)"
          className="border p-2 w-full h-32 mb-2"
        />
        <button
          onClick={addItems}
          disabled={loadingActions['addItems']}
          className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 transition-colors flex items-center"
        >
          {loadingActions['addItems'] ? (
            <><Loader2 className="animate-spin mr-2" size={18} /> Adding...</>
          ) : (
            'Add Items'
          )}
        </button>
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Search Items</h2>
        <div className="flex mb-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter search query"
            className="border p-2 mr-2 flex-grow"
          />
          <input
            type="text"
            value={namespace}
            onChange={(e) => setNamespace(e.target.value)}
            placeholder="Enter namespace (optional)"
            className="border p-2 mr-2"
          />
          <button
            onClick={searchItems}
            disabled={loadingActions['searchItems']}
            className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600 transition-colors flex items-center"
          >
            {loadingActions['searchItems'] ? (
              <><Loader2 className="animate-spin mr-2" size={18} /> Searching...</>
            ) : (
              'Search'
            )}
          </button>
        </div>
        {searchResults.length > 0 && (
          <div className="bg-gray-100 p-4 rounded">
            <h3 className="font-semibold mb-2">Search Results:</h3>
            <ul className="list-disc pl-5">
              {searchResults.map((result, index) => (
                <li key={index}>{result.text} (Score: {result.score.toFixed(4)})</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Delete All Items</h2>
        <button
          onClick={deleteAllItems}
          disabled={loadingActions['deleteAllItems']}
          className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors flex items-center"
        >
          {loadingActions['deleteAllItems'] ? (
            <><Loader2 className="animate-spin mr-2" size={18} /> Deleting...</>
          ) : (
            'Delete All Items from Selected Index'
          )}
        </button>
      </div>

      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Delete Items from Namespace</h2>
        <div className="flex">
          <input
            type="text"
            value={deleteNamespace}
            onChange={(e) => setDeleteNamespace(e.target.value)}
            placeholder="Enter namespace to delete"
            className="border p-2 mr-2 flex-grow"
          />
          <button
            onClick={deleteNamespaceItems}
            disabled={loadingActions['deleteNamespaceItems']}
            className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600 transition-colors flex items-center"
          >
            {loadingActions['deleteNamespaceItems'] ? (
              <><Loader2 className="animate-spin mr-2" size={18} /> Deleting...</>
            ) : (
              'Delete Items from Namespace'
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default PineconeManager;
