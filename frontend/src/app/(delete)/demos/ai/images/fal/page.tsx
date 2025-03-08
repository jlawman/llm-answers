'use client';

import React, { useState } from 'react';
import { Heading, Subheading } from '@/components/ui/heading';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import Lightbox from '@/components/ui/lightbox';
import { Loader2 } from 'lucide-react';

export default function FalImageGenerationDemo() {
  const [prompt, setPrompt] = useState('');
  const [generatedImage, setGeneratedImage] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/fal-flux-1-1-pro', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const result = await response.json();
      setGeneratedImage(result.data.images[0].url);
    } catch (error) {
      console.error('Error generating image:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
      <Heading className="text-center mb-8">Fal AI Image Generation</Heading>
      <Subheading className="text-center mb-12">https://fal.ai/models/fal-ai/flux-pro/v1.1/api </Subheading>

      <div className="space-y-6">
        <Input
          type="text"
          placeholder="Enter your image prompt here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full"
        />
        <Button
          onClick={handleGenerate}
          disabled={isLoading || !prompt}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            'Generate Image'
          )}
        </Button>
      </div>

      {generatedImage && (
        <div className="mt-8">
          <Lightbox
            src={generatedImage}
            alt="Generated image"
            width={512}
            height={512}
          />
        </div>
      )}
    </div>
  );
}
