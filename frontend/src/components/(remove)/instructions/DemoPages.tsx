import React from 'react';
import { Button } from '@/components/ui/button'
import { Container } from '@/components/ui/container'

export function DemoPages() {
    const demoCategories = [
      {
        name: 'UI',
        demos: [
          { name: 'Marketing Landing Page', href: '/marketing', description: 'Explore a sample marketing landing page' },
        ],
      },
      {
        name: 'LLM',
        demos: [
          { name: 'LLM API Demo', href: '/ai/llms', description: 'Test out LLM API integration' },
          { name: 'Vector DB Demo', href: '/ai/vectordb', description: 'See vector database in action' },
        ],
      },
      {
        name: 'Images',
        demos: [
          { name: 'Flux 1.1 Pro', href: '/ai/images/fal', description: 'Generate images with Fal AI' },
        ],
      },
      {
        name: 'Extras',
        demos: [
          { name: 'Toast Demo', href: '/extras/toast', description: 'View toast notification examples' },
        ],
      },
      {
        name: 'Monitoring',
        demos: [
          { name: 'Sentry Demo Error', href: '/monitoring/sentry-example-page', setupHref: '/setup/sentry', description: 'Trigger and monitor Sentry errors' },
          { name: 'Fathom Demo Events', href: '/monitoring/log-fathom-events', description: 'Code to trigger fathom events' },
        ],
      },
      {
        name: 'Users',
        demos: [
          { name: 'Add Clerk', setupHref: '/setup/clerk', description: 'Set up Clerk for user authentication' },
        ],
      },
      {
        name: 'Storage',
        demos: [
          { name: 'Vercel Edge Config', href: '/storage/vercel-edge', description: 'See how to use Vercel Edge Config' },
          { name: 'Neon + Drizzle with ClerkID', href: '/storage/drizzle-neon', description: 'See how to use Neon with Clerk' },
        ],
      },
    ] as const;
  
    type Demo = {
      name: string;
      href?: string;
      setupHref?: string;
      description: string;
    };
  
    return (
      <Container className="py-16 sm:py-24">
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl mb-12">
          Demos
        </h1>
        <div className="space-y-16">
          {demoCategories.map((category, categoryIndex) => (
            <section key={categoryIndex}>
              <h2 className="text-3xl font-bold text-gray-900 mb-8">{category.name}</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {category.demos.map((demo: Demo, demoIndex) => (
                  <div key={demoIndex} className="bg-white p-6 rounded-lg shadow-md">
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">
                      {demo.name}
                    </h3>
                    <p className="mb-4 text-gray-600">{demo.description}</p>
                    <div className="flex space-x-4">
                      {demo.href && (
                        <Button href={demo.href}>Demo</Button>
                      )}
                      {demo.setupHref && (
                        <Button href={demo.setupHref} variant="secondary">Setup</Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      </Container>
    )
  }
  