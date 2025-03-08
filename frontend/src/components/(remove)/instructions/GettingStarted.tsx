import React, { useState } from 'react';
import { Button } from '@/components/ui/button'
import { Container } from '@/components/ui/container'
import { Checkbox, CheckboxField } from '@/components/ui/checkbox'
import { Loader2 } from 'lucide-react';
// NOTE: Eventually the guides should be in accordian on this page. (not fully necessary)
// Or should be able to open all essential items at once in new tabs.

export function GettingStarted() {
  const setupCategories = [
    {
      name: 'Essentials',
      items: [
        { function: 'Repository', name: 'GitHub', href: '/setup/github', description: 'Set up your repository' },
        { function: 'Deployment', name: 'Vercel', href: '/setup/vercel', description: 'Set up your deployment pipeline' },
        { function: 'Error Tracking', name: 'Sentry', href: '/setup/sentry', description: 'Configure error tracking and monitoring' },
        { function: 'Analytics', name: 'Fathom', href: '/setup/fathom', description: 'Set up privacy-friendly analytics' },
        { function: 'LLM Monitor', name: 'Braintrust', href: '/setup/braintrust', description: 'Integrate AI model evaluation' },
        { function: 'Authentication', name: 'Clerk', href: '/setup/clerk', description: 'Implement user authentication' },
      ],
    },
    {
      name: 'Optional',
      items: [
        { function: 'Vector Database', name: 'Pinecone', href: '/setup/pinecone', description: 'Integrate vector database for AI' },
        { function: 'Payment Processing', name: 'Stripe', href: '/setup/stripe', description: 'Set up payment processing' },
      ],
    },
    {
      name: 'Models',
      items: [
        { function: 'LLM', name: 'OpenAI', href: '/setup/openai', description: 'Set up OpenAI API key' },
        { function: 'LLM', name: 'Anthropic', href: '/setup/anthropic', description: 'Set up Anthropic API key' },
        { function: 'LLM', name: 'Groq', href: '/setup/groq', description: 'Set up Groq API key' },
        { function: 'Images', name: 'Fal', href: '/setup/fal', description: 'Set up Fal API key' },
      ],
    },
  ] as const;

  const [checkedItems, setCheckedItems] = useState<Record<string, boolean>>({});
  const [loadingItem, setLoadingItem] = useState<string | null>(null);

  const handleCheckboxChange = (itemName: string) => {
    setCheckedItems(prev => ({ ...prev, [itemName]: !prev[itemName] }));
  };

  const handleReset = () => {
    setCheckedItems({});
  };

  const handleSetup = (itemName: string, href: string) => {
    setLoadingItem(itemName);
    window.location.href = href;
  };

  const handleOpenAllEssentials = () => {
    const essentialItems = setupCategories.find(category => category.name === 'Essentials')?.items || [];
    essentialItems.forEach(item => {
      window.open(item.href, '_blank', 'noopener,noreferrer');
    });
  };

  return (
    <Container className="py-16 sm:py-24">
      <div className="mb-8">
        <Button onClick={handleOpenAllEssentials}>
          Open All Essential Guides
        </Button>
      </div>
      
      <div className="space-y-16">
        {setupCategories.map((category, categoryIndex) => (
          <section key={categoryIndex}>
            <h2 className="text-3xl font-bold text-gray-900 mb-8">{category.name}</h2>
            <div className="space-y-6">
              {category.items.map((item, itemIndex) => (
                <CheckboxField key={itemIndex}>
                  <Checkbox
                    checked={checkedItems[item.name] || false}
                    onChange={() => handleCheckboxChange(item.name)}
                  />
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">
                      {item.function} <span className="font-normal">({item.name})</span>
                    </h3>
                    <p className="mt-1 text-gray-600"></p>
                    <Button
                      onClick={() => handleSetup(item.name, item.href)}
                      disabled={loadingItem === item.name}
                      className="mt-2"
                    >
                      {loadingItem === item.name ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Loading...
                        </>
                      ) : (
                        'Setup'
                      )}
                    </Button>
                  </div>
                </CheckboxField>
              ))}
            </div>
          </section>
        ))}
      </div>
      
      <div className="mt-12 flex justify-between items-center">
        <Button onClick={handleReset} variant="outline">
          Reset Checklist
        </Button>
        <Button onClick={handleOpenAllEssentials}>
          Open All Essential Guides
        </Button>
      </div>
    </Container>
  );
}
