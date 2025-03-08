'use client';

import React, { useState } from 'react';
import { Dialog, DialogTitle, DialogBody, DialogActions } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Loader2, Download, RefreshCw, Wand2 } from 'lucide-react';
import Image from 'next/image';

const ASPECT_RATIOS = [
  { value: 'square_hd', label: 'Square HD' },
  { value: 'square', label: 'Square' },
  { value: 'portrait_4_3', label: 'Portrait 4:3' },
  { value: 'portrait_16_9', label: 'Portrait 16:9' },
  { value: 'landscape_4_3', label: 'Landscape 4:3' },
  { value: 'landscape_16_9', label: 'Landscape 16:9' },
  { value: 'custom_900x300', label: '900 x 300' },
];

export function LogoGenerationModal({ prompt }: { prompt: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const [aspectRatio, setAspectRatio] = useState('square_hd');
  const [isLoading, setIsLoading] = useState(false);
  const [generatedLogo, setGeneratedLogo] = useState<string | null>(null);

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const imageSize = aspectRatio === 'custom_900x300' 
        ? { width: 900, height: 300 } 
        : aspectRatio;

      const response = await fetch('/api/fal-flux-1-1-pro', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `${prompt}.`,
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
      if (result.data?.images?.[0]) {
        const image = result.data.images[0];
        setGeneratedLogo(typeof image === 'string' ? image : image.url);
      }
    } catch (error) {
      console.error('Error generating logo:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!generatedLogo) return;
    try {
      const response = await fetch(generatedLogo);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      link.download = 'generated-logo.jpg';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Error downloading logo:', error);
    }
  };

  return (
    <>
      <Button size="sm" variant="outline" className="gap-2" onClick={() => setIsOpen(true)}>
        <Wand2 className="h-4 w-4" />
        Generate Image
      </Button>

      <Dialog open={isOpen} onClose={() => setIsOpen(false)}>
        <DialogTitle>
          Generate Logo Image
        </DialogTitle>
        
        <DialogBody>
          <div className="space-y-4">
            <select
              value={aspectRatio}
              onChange={(e) => setAspectRatio(e.target.value)}
              className="w-full p-3 mb-4 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {ASPECT_RATIOS.map((ratio) => (
                <option key={ratio.value} value={ratio.value}>
                  {ratio.label}
                </option>
              ))}
            </select>

            {generatedLogo && (
              <div className="relative mt-4">
                <Image
                  src={generatedLogo}
                  alt="Generated logo"
                  width={900}
                  height={900}
                  className="mx-auto rounded-lg max-w-full h-auto"
                />
                <div className="absolute bottom-2 right-2 flex gap-2">
                  <Button size="sm" variant="secondary" onClick={handleDownload}>
                    <Download className="h-4 w-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleGenerate}
                    disabled={isLoading}
                  >
                    <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  </Button>
                </div>
              </div>
            )}
          </div>
        </DialogBody>

        <DialogActions>
          <Button 
            variant="outline" 
            onClick={() => setIsOpen(false)}
          >
            Close
          </Button>
          
          <Button 
            onClick={handleGenerate}
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              'Generate'
            )}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
} 