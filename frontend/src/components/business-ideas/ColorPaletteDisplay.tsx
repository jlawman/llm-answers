import React, { useState } from 'react';
import { CheckIcon } from '@heroicons/react/24/outline';

type ColorInfo = {
  name: string;
  hex: string;
};

type ColorPaletteDisplayProps = {
  colors?: ColorInfo[];
  explanation?: string;
};

export function ColorPaletteDisplay({ colors = [], explanation = '' }: ColorPaletteDisplayProps) {
  const [copiedColor, setCopiedColor] = useState<string | null>(null);

  const handleColorClick = async (hex: string) => {
    try {
      await navigator.clipboard.writeText(hex);
      setCopiedColor(hex);
      setTimeout(() => setCopiedColor(null), 2000); // Reset after 2 seconds
    } catch (err) {
      console.error('Failed to copy color:', err);
    }
  };

  if (!colors.length) {
    return <div>No color palette available</div>;
  }

  return (
    <div className="space-y-8">
      {/* Color Swatches */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        {colors.map((color) => (
          <div
            key={color.hex}
            className="flex flex-col space-y-2 group hover:scale-105 transition-transform duration-200"
          >
            <div
              className="h-24 rounded-lg shadow-md cursor-pointer relative overflow-hidden"
              style={{ backgroundColor: color.hex }}
              onClick={() => handleColorClick(color.hex)}
            >
              <div 
                className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 rounded-lg"
                style={{ 
                  backgroundColor: `${color.hex}dd`,
                }}
              >
                {copiedColor === color.hex ? (
                  <div className="flex items-center space-x-2 text-white animate-fade-in">
                    <CheckIcon className="h-5 w-5" />
                    <span className="font-mono text-sm">Copied!</span>
                  </div>
                ) : (
                  <span className="text-white font-mono drop-shadow-md">{color.hex}</span>
                )}
              </div>
            </div>
            <span className="text-sm font-medium text-gray-700">{color.name}</span>
          </div>
        ))}
      </div>

      {/* Color Explanation */}
      {explanation && (
        <div className="prose max-w-none">
          <h3 className="text-xl font-semibold mb-4">Color Palette Explanation</h3>
          <div className="bg-gray-50 rounded-lg p-6">
            {explanation.split('\n\n').map((paragraph, index) => (
              <p key={index} className="mb-4 last:mb-0">
                {paragraph}
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
} 