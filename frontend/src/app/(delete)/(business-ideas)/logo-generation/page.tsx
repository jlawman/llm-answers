'use client';

import React, { useState } from 'react';
import { Container } from '@/components/ui/container';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Loader2, Wand2, Download, RefreshCw } from 'lucide-react';
import { Fieldset } from '@/components/ui/fieldset';
import Image from 'next/image';

export default function LogoGenerationPage() {
  const [prompt, setPrompt] = useState('');
  const [generatedLogos, setGeneratedLogos] = useState<(string | { url: string })[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isEnriching, setIsEnriching] = useState(false);
  const [aspectRatio, setAspectRatio] = useState<string>('square_hd');

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const imageSize = aspectRatio === 'custom_900x300' 
        ? { width: 900, height: 300 } 
        : aspectRatio;

      console.log(imageSize);
      const response = await fetch('/api/fal-flux-1-1-pro', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `Generate a logo based on this description: ${prompt}.`,
          image_size: imageSize,
          num_inference_steps: 28,
          guidance_scale: 3.5,
          num_images: 1,
          enable_safety_checker: true,
          strength: 1,
          output_format: "jpeg"
        }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error || 'An error occurred');
      }
      if (result.data && result.data.images && Array.isArray(result.data.images)) {
        setGeneratedLogos(result.data.images.map((img: string | { url: string }) => 
          typeof img === 'string' ? img : img.url
        ));
      } else {
        console.error('Unexpected response format:', result);
      }
    } catch (error) {
      console.error('Error generating logos:', error);
      // You might want to set an error state here and display it to the user
    } finally {
      setIsLoading(false);
    }
  };

  const handleEnrichPrompt = async () => {
    setIsEnriching(true);
    try {
      const response = await fetch('/api/ask-groq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: 'Create an elegant prompt for designing a logo in image generation. @logo @design',
        }),
      });
      const data = await response.json();
      if (data.result) {
        setPrompt(prev => `${prev} ${data.result}`);
      } else {
        console.error('No result returned from ask-groq');
      }
    } catch (error) {
      console.error('Error enriching prompt:', error);
    } finally {
      setIsEnriching(false);
    }
  };

  const handleDownload = async (imageUrl: string, index: number) => {
    try {
      // Fetch the image data
      const response = await fetch(imageUrl);
      const blob = await response.blob();

      // Create a temporary URL for the blob
      const blobUrl = window.URL.createObjectURL(blob);

      // Create a link element and trigger the download
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = `generated-logo-${index + 1}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up the temporary URL
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Error downloading logo:', error);
      // You might want to show an error message to the user here
    }
  };

  const handleReroll = async (index: number) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/fal-flux-1-1-pro', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `Generate a logo based on this description: ${prompt}. The logo should be square-shaped.`,
          image_size: "square",
          num_inference_steps: 28,
          guidance_scale: 3.5,
          num_images: 1,
          enable_safety_checker: true,
          strength: 1,
          output_format: "jpeg"
        }),
      });
      const result = await response.json();
      setGeneratedLogos(prevLogos => {
        const newLogos = [...prevLogos];
        newLogos[index] = result.images[0];
        return newLogos;
      });
    } catch (error) {
      console.error('Error rerolling logo:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container className="py-8">
      <h1 className="text-3xl font-bold mb-8">Logo Generator</h1>

      <p className="text-gray-600 text-sm mb-12">
        Describe the logo you want, optionally use the ✨ wand to enhance your description with professional design terms, 
        then click Generate to create a logo for your SaaS.
      </p>

      <div className="space-y-6">
        <div className="flex flex-col space-y-4">
          <Textarea
            placeholder="Describe your logo..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="flex-grow"
            rows={4}
          />
          <Fieldset>
            <select
              className="w-full p-2 mb-4 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              value={aspectRatio}
              onChange={(e) => setAspectRatio(e.target.value)}
            >
              <option value="square">Square</option>
              <option selected value="square_hd">Square HD</option>
              <option value="portrait_4_3">Portrait 4:3</option>
              <option value="portrait_16_9">Portrait 16:9</option>
              <option value="landscape_4_3">Landscape 4:3</option>
              <option value="landscape_16_9">Landscape 16:9</option>
              <option value="custom_900x300">900 x 300</option>
            </select>
            <Button onClick={handleEnrichPrompt} className="mt-4flex-shrink-0" disabled={isEnriching}>
              {isEnriching ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Wand2 className="mr-2 h-4 w-4" />
              )}
              Enrich
            </Button>
          </Fieldset>
        </div>
        <Button
          onClick={handleGenerate}
          disabled={isLoading || prompt.trim() === ''}
          className="w-full"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : (
            'Generate Logo'
          )}
        </Button>
      </div>

      {Array.isArray(generatedLogos) && generatedLogos.length > 0 && (
        <div className="mt-8 grid grid-cols-2 gap-4">
          {generatedLogos.map((logo, index) => (
            <div key={index} className="relative">
              {logo && (
                <>
                  <Image
                    src={typeof logo === 'string' ? logo : logo.url}
                    alt={`Generated logo ${index + 1}`}
                    width={512}
                    height={512}
                    className="mx-auto rounded-lg max-w-full h-auto"
                    unoptimized
                  />
                  <div className="absolute bottom-2 right-2 flex space-x-2">
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => handleDownload(typeof logo === 'string' ? logo : logo.url, index)}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                    variant="secondary"
                      onClick={() => handleReroll(index)}
                      disabled={isLoading}
                    >
                      <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                    </Button>
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </Container>
  );
}
