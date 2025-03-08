/////////////
// Subject Matter: SaaS Brand Ideation & Strategy
// Tone: Professional, innovative, empowering
// Features:
// - AI-powered business idea generation
// - Brand strategy development
// - Visual identity creation
/////////////

// Writing instructions:
// 1. Emphasize transformation from idea to brand
// 2. Highlight intelligent digital tools
// 3. Maintain professional yet approachable tone

import type { Metadata } from 'next'
import { Footer } from '@/components/ui/footer'
import { Hero } from '@/components/marketing/Hero'
export const metadata: Metadata = {
  description:
    'Free tools to help you brainstorm and generate creative business ideas.',
}

export default function Home() {
  return (
    <div className="overflow-hidden">
      <Hero 
        show={true}
        bannerText="Need business ideas?" 
        bannerLink="/business-ideas" 
        heading="Never Run Out of SaaS Ideas" 
        subheading="Daily new SaaS ideas and brand definition tool to assist rapid conceptualization and prototyping." 
        theme="A"
        variant="withoutLink"
      />
      <main>
      </main>

      <Footer />
    </div>
  )
}
