'use client';

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Loader2, Copy, Check } from 'lucide-react';
import { toast } from 'sonner';
import { Tabs } from "@/components/ui/tabs";
import Markdown from 'react-markdown';
import { Container } from '@/components/ui/container';
import { LogoRemixButton } from '@/components/business-ideas/LogoRemixButton'
import { LogoGenerationModal } from '@/components/business-ideas/LogoGenerationModal';
import { ColorPaletteDisplay } from '@/components/business-ideas/ColorPaletteDisplay';
import { NameRemixButton } from '@/components/business-ideas/NameRemixButton';
import { motion, AnimatePresence } from 'framer-motion'

// Define types for better type safety
type BrandSection = {
  id: string;
  title: string;
  buttonText: string;
  promptTemplate: string;
  description: string;
};

// Update the type to include logo descriptions
type LogoResults = {
  logo_name_1?: string;
  logo_description_1: string;
  logo_name_2?: string;
  logo_description_2: string;
  logo_name_3?: string;
  logo_description_3: string;
};

// Add new type definition
type ColorPaletteResult = {
  colors: { name: string; hex: string; }[];
  explanation: string;
};

const BRAND_SECTIONS: BrandSection[] = [
  {
    id: 'brandSurvey',
    title: 'Brand Strategy',
    buttonText: 'Step 1: Analyze Brand Strategy',
    promptTemplate: 'brandSurveyPrompt',
    description: '🎯 Start here! Define your core brand strategy - this is required before accessing other brand tools.'
  },
  {
    id: 'logoIdeas',
    title: 'Logo Ideas & Images',
    buttonText: 'Generate Logo Ideas',
    promptTemplate: 'brandLogoPrompt',
    description: 'Generate creative logo ideas and downloadable logos that align with your brand identity and visual style.'
  },
  {
    id: 'brandName',
    title: 'Business Name and Domains',
    buttonText: 'Generate Names',
    promptTemplate: 'brandNamePrompt',
    description: 'Create a memorable, domain-friendly name that reflects your SaaS solution and resonates with your target market.'
  },
  {
    id: 'brandColors',
    title: 'Brand Colors',
    buttonText: 'Generate Color Palette',
    promptTemplate: 'brandColorsPrompt',
    description: 'Get a modern, tech-focused color palette that builds trust and professionalism in the SaaS space.'
  },
  {
    id: 'brandGuidelines',
    title: 'Brand Guidelines',
    buttonText: 'Generate Guidelines',
    promptTemplate: 'brandGuidelinesPrompt',
    description: 'Create comprehensive guidelines for your SaaS brand, including voice, tone, and visual elements for web and product interfaces. Save to a Markdown file and tag in Cursor (or other coding tool) to help design your marketing pages or Saas Components.'
  },
];

export default function BrandPage() {
  const [input, setInput] = useState('');
  const [results, setResults] = useState<Record<string, string | string[] | LogoResults | ColorPaletteResult | undefined>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<string>('');
  const [currentInput, setCurrentInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [shouldAnimate, setShouldAnimate] = useState(false)

  // Add new ref for results section
  const resultsRef = React.useRef<HTMLDivElement>(null);

  // Add function to check if brand survey is completed
  const isBrandSurveyCompleted = () => {
    return results['brandSurvey'] !== undefined;
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (e.target.value !== currentInput) {
      setResults(prev => ({ ...prev, brandSurvey: undefined }));
      setActiveTab(''); // Clear active tab when input changes
    }
  };

  const generateContent = async (section: BrandSection) => {
    if (!input.trim()) {
      toast.error('Please enter a description first');
      return;
    }

    // Check if trying to access non-survey section without completing survey
    if (section.id !== 'brandSurvey' && !isBrandSurveyCompleted()) {
      toast.error('Please complete the Brand Survey first');
      return;
    }

    setLoading(prev => ({ ...prev, [section.id]: true }));

    try {
      const response = await fetch('/api/ask-anthropic-template', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: section.id === 'brandSurvey' ? input : results['brandSurvey'],
          promptTemplateName: section.promptTemplate,
        }),
      });

      if (!response.ok) throw new Error('API request failed');

      const data = await response.json();
      let result = data.result || data;
      
      console.log('Raw result:', result); // Debug log

      // Special handling for logo ideas and brand colors
      if (section.id === 'logoIdeas') {
        result = data.result || data;
      } else if (section.id === 'brandColors') {
        // Get the color_palette content from the result
        const colorPaletteContent = typeof result === 'string' 
          ? result 
          : result.color_palette || JSON.stringify(result);
        
        console.log('Color palette content:', colorPaletteContent); // Debug log
        
        // Extract individual colors
        const colors = [
          { regex: /<primary_color>(#[A-Fa-f0-9]{6})<\/primary_color>/, name: 'Primary' },
          { regex: /<secondary_color1>(#[A-Fa-f0-9]{6})<\/secondary_color1>/, name: 'Secondary 1' },
          { regex: /<secondary_color2>(#[A-Fa-f0-9]{6})<\/secondary_color2>/, name: 'Secondary 2' },
          { regex: /<accent_color1>(#[A-Fa-f0-9]{6})<\/accent_color1>/, name: 'Accent 1' },
          { regex: /<accent_color2>(#[A-Fa-f0-9]{6})<\/accent_color2>/, name: 'Accent 2' },
        ].map(({ regex, name }) => ({
          name,
          hex: colorPaletteContent.match(regex)?.[1] || '#000000',
        }));

        // Extract explanation using a more compatible regex pattern
        const explanation = colorPaletteContent.match(/<color_explanation>([^]*?)<\/color_explanation>/)?.[1]?.trim() || '';

        result = { colors, explanation };
        console.log('Parsed result:', result); // Debug log
      }

      if (section.id === 'brandSurvey') {
        setCurrentInput(input);
      }

      setResults(prev => ({
        ...prev,
        [section.id]: result,
      }));
      
      setActiveTab(section.title);

      // Add smooth scroll to results after generation
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);

    } catch (error) {
      console.error('Error in generateContent:', error);
      toast.error(`Failed to generate ${section.title.toLowerCase()}`);
    } finally {
      setLoading(prev => ({ ...prev, [section.id]: false }));
    }
  };

  const copyToClipboard = (text: string, sectionId: string) => {
    navigator.clipboard.writeText(text).then(
      () => {
        toast.success('Copied to clipboard');
        setCopiedId(sectionId);
        setTimeout(() => setCopiedId(null), 2000); // Reset after 2 seconds
      },
      () => toast.error('Failed to copy')
    );
  };

  return (
    <Container className="py-8">
      <h1 className="text-3xl font-bold mb-8">Brand Development</h1>

      {/* Simplified initial input section */}
      <div className="mb-12">
        <h2 className="text-2xl font-semibold mb-4">Describe Your SaaS</h2>
        <textarea
          value={input}
          onChange={handleInputChange}
          placeholder="Describe your SaaS product, target market, and core problem you&apos;re solving..."
          className="w-full p-3 mb-4 min-h-[100px] border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
        />
        <p className="text-sm text-gray-600 mb-4">
          Start by describing your product. Once brand survey answers are generated, you&apos;ll unlock access to all brand development tools below.
        </p>
        <Button
          onClick={() => generateContent(BRAND_SECTIONS[0])}
          disabled={loading['brandSurvey'] || !input.trim()}
          className="w-full md:w-auto"
        >
          {loading['brandSurvey'] ? (
            <div className="flex items-center">
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Answering brand survey questions...
            </div>
          ) : (
            'Define Brand'
          )}
        </Button>
      </div>

      {/* Brand Generation Tools Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {BRAND_SECTIONS.slice(1).map((section) => (
          <div 
            key={section.id} 
            className={`bg-white p-6 rounded-lg shadow-md ${
              !isBrandSurveyCompleted() ? 'opacity-50' : ''
            }`}
          >
            <h3 className="text-xl font-bold mb-2">{section.title}</h3>
            <p className="text-gray-600 text-sm mb-4">{section.description}</p>
            <Button
              onClick={() => generateContent(section)}
              disabled={
                loading[section.id] || 
                !isBrandSurveyCompleted() ||
                input !== currentInput
              }
              className="w-full"
              variant="secondary"
            >
              {loading[section.id] ? (
                <div className="flex items-center">
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Generating...
                </div>
              ) : (
                section.buttonText
              )}
            </Button>
          </div>
        ))}
      </div>

      {/* Results Display */}
      {Object.keys(results).length > 0 && activeTab && (
        <>
          {/* Add scroll indicator for mobile */}
          <div className="md:hidden flex flex-col items-center mt-6 animate-bounce">
            <div className="text-sm text-gray-600">Scroll to view results</div>
            <svg 
              className="w-6 h-6 text-gray-600" 
              fill="none" 
              strokeWidth="2" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path d="M19 14l-7 7m0 0l-7-7m7 7V3"></path>
            </svg>
          </div>

          {/* Add ref to results section */}
          <div ref={resultsRef} className="mt-8">
            <Tabs
              tabs={Object.keys(results).map(sectionId => ({
                name: BRAND_SECTIONS.find(s => s.id === sectionId)?.title || '',
                current: BRAND_SECTIONS.find(s => s.id === sectionId)?.title === activeTab
              }))}
              onChange={(selectedTab) => {
                setActiveTab(selectedTab.name);
              }}
            />
            
            {Object.entries(results).map(([sectionId, content]) => {
              const section = BRAND_SECTIONS.find(s => s.id === sectionId);
              if (!section || section.title !== activeTab) return null;
              
              if (section.id === 'logoIdeas') {
                const logoContent = content as LogoResults;
                return (
                  <div key={sectionId} className="mt-4 space-y-4">
                    {[1, 2, 3].map((num) => {
                      const nameKey = `logo_name_${num}` as keyof LogoResults;
                      const descKey = `logo_description_${num}` as keyof LogoResults;
                      const description = logoContent[descKey];
                      const name = logoContent[nameKey] || `Logo Concept ${num}`;

                      if (!description) return null;

                      return (
                        <div key={nameKey} className="prose max-w-none bg-white rounded-lg shadow-md p-6 relative">
                          <div className="flex gap-2 absolute top-4 right-4">
                            <LogoGenerationModal prompt={description} />
                            <LogoRemixButton 
                              originalDescription={description}
                              onRemix={(newDescription) => {
                                const updatedLogoContent = { ...logoContent };
                                updatedLogoContent[descKey] = newDescription;
                                setResults(prev => ({
                                  ...prev,
                                  [sectionId]: updatedLogoContent
                                }));
                              }}
                            />
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => copyToClipboard(description, `${sectionId}-${descKey}`)}
                            >
                              {copiedId === `${sectionId}-${descKey}` ? (
                                <Check className="h-4 w-4" />
                              ) : (
                                <Copy className="h-4 w-4" />
                              )}
                            </Button>
                          </div>
                          <h3>{name}</h3>
                          <Markdown>{description}</Markdown>
                        </div>
                      );
                    })}
                  </div>
                );
              } else if (section.id === 'brandColors' && content && typeof content === 'object' && 'colors' in content) {
                const colorContent = content as ColorPaletteResult;
                return (
                  <div key={sectionId} className="mt-4">
                    <ColorPaletteDisplay colors={colorContent.colors} explanation={colorContent.explanation} />
                  </div>
                );
              } else if (section.id === 'brandName') {
                return (
                  <div key={sectionId} className="mt-4">
                    <AnimatePresence mode="wait">
                      <motion.div 
                        key={content as string}
                        initial={shouldAnimate ? { opacity: 0, y: 20 } : false}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                        transition={{ duration: 0.5 }}
                        className="prose max-w-none bg-white rounded-lg shadow-md p-6 relative"
                      >
                        <div className="flex gap-2 absolute top-4 right-4">
                          <NameRemixButton
                            originalNames={[content as string]}
                            onNewNames={(newNames, animate) => {
                              setShouldAnimate(!!animate)
                              setResults(prev => ({
                                ...prev,
                                [sectionId]: newNames[0]
                              }))
                            }}
                            brandSurvey={results['brandSurvey'] as string}
                          />
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => content && copyToClipboard(content.toString(), sectionId)}
                          >
                            {copiedId === sectionId ? (
                              <Check className="h-4 w-4" />
                            ) : (
                              <Copy className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                        <Markdown>{typeof content === 'string' ? content : ''}</Markdown>
                      </motion.div>
                    </AnimatePresence>
                  </div>
                );
              }
              
              // Default handling for other sections
              return (
                <div key={sectionId} className="mt-4">
                  <div className="prose max-w-none bg-white rounded-lg shadow-md p-6 relative">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => content && copyToClipboard(content.toString(), sectionId)}
                      className="absolute top-4 right-4"
                    >
                      {copiedId === sectionId ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                    <Markdown>{typeof content === 'string' ? content : ''}</Markdown>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </Container>
  );
}
