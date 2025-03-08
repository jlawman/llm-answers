/////////////
// Subject Matter: Icon Creator
// Tone: Playful, creative, and imaginative
// Features:
// - Create icons from text
// - Choose from a variety of styles
// - Download in SVG format
/////////////

// Writing instructions:
// 1. Coherently write about the subject matter
// 2. Follow the speicfied tone.
// 3. Follow the component directions for use

// Themes:
// A. Trendy and Modern
// B. Minimalist and Elegant


import { Testimonials } from '@/components/ui/testimonials'
import type { Metadata } from 'next'
import { Footer } from '@/components/ui/footer'
import { Hero } from '@/components/marketing/Hero'
export const metadata: Metadata = {
  description:
    'Objective AI analysis of anything you have wanted to know the truth about.',
}


export default function Home() {
  return (
    <div className="overflow-hidden">
      <Hero 
        show={true} // Boolean to control visibility of the Hero component
        bannerText="[Concise, attention-grabbing text for the banner]" // Short, impactful message to highlight current status or offering
        bannerLink="/example" // Link to the next step in the user journey
        heading="[Powerful, brief main headline]" // Clear, compelling main message in 2-4 words
        subheading="[Expanded value proposition]" // 1-2 sentences explaining the core benefit to the user
        theme="A"
        variant="withLink" // Optional: 'withLink' or 'withoutLink' // Determines if banner link is used
      />
      <main>
      </main>
      <Testimonials />
      <Footer />
    </div>
  )
}
