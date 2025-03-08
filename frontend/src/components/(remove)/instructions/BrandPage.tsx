import React from 'react';
import { Button } from '@/components/ui/button'
import { Container } from '@/components/ui/container'

export function BrandPage() {
    const brandPages = [
      {
        name: 'Brand Overview',
        href: '/brand',
        description: 'View and manage your brand assets',
      },
      {
        name: 'Logo Generator',
        href: '/brand/logo',
        description: 'Generate and customize logos for your brand',
      },
      {
        name: 'Business Ideas',
        href: '/brand/business-ideas',
        description: 'View AI-generated business ideas and opportunities',
      }
    ];
  
    return (
      <Container className="py-16 sm:py-24">
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl mb-12">
          Brand Management
        </h1>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {brandPages.map((page, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md">
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                {page.name}
              </h3>
              <p className="mb-4 text-gray-600">{page.description}</p>
              <Button href={page.href}>Go to {page.name}</Button>
            </div>
          ))}
        </div>
      </Container>
    )
  }