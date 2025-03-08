import { db } from '@/app/db'
import { UserGeneratedBusinessIdeas } from '@/app/db/schema'
import { NextResponse } from 'next/server'
import { v4 as uuidv4 } from 'uuid'
import { Anthropic } from '@anthropic-ai/sdk'
import { desc } from 'drizzle-orm'
import { ProductHuntEntries } from '@/app/db/schema'

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

export async function POST(request: Request) {
  try {
    const { message } = await request.json()
    
    // Get recent ProductHunt entries for inspiration
    const recentProducts = await db.select({
      name: ProductHuntEntries.name,
      description: ProductHuntEntries.description
    })
    .from(ProductHuntEntries)
    .limit(10)
    .orderBy(desc(ProductHuntEntries.createTs))

    const productHuntInspirationText = recentProducts.map(product => 
      `- ${product.name}: ${product.description}`
    ).join('\n')

    // Extract industry and group from the message
    const industryMatch = message.match(/for the (.*?) industry in the (.*?) sector/)
    const industry = {
      industry: industryMatch?.[1] || '',
      group: industryMatch?.[2] || ''
    }

    // Randomization options with logical groupings
    const targetMarket = ['B2B', 'B2C', 'B2B2C'].at(Math.floor(Math.random() * 3))

    // const businessScale = (() => {
    //   // B2B typically needs more initial investment
    //   if (targetMarket === 'B2B') {
    //     return ['venture-backed startup', 'bootstrapped startup'].at(Math.floor(Math.random() * 2))
    //   }
    //   return ['bootstrapped startup', 'venture-backed startup', 'side project'].at(Math.floor(Math.random() * 3))
    // })()

    const specialFocus = (() => {
      const options = [
        'voice assistant capabilities',
        'multimodal AI integration',
        'real-time processing',
        'llms',
        'forecasting',
        'speech synthesis',
        'speech recognition',
        'image analysis',
        'document analysis',
        'automation',
        'workflow optimization',
        'natural language processing',
        'document analysis',
      ]
      return options.at(Math.floor(Math.random() * options.length))
    })()

    const techFocus = (() => {
      const baseOptions = [
        'AI-powered automation and workflow optimization',
        'advanced language models for business communication',
        'predictive analytics and forecasting',
        'computer vision and visual recognition',
        'natural language processing for document analysis',
        'intelligent process automation'
      ]
      
      return baseOptions.at(Math.floor(Math.random() * baseOptions.length))
    })()

    // Development timeline based on business scale
    // const developmentTimeline = (() => {
    //   if (businessScale === 'venture-backed startup') {
    //     return ['6-month development cycle', 'year-long development'].at(Math.floor(Math.random() * 2))
    //   }
    //   return ['quick 3-month MVP', 'rapid 1-month prototype'].at(Math.floor(Math.random() * 2))
    // })()

    // const targetGeography = (() => {
    //   if (businessScale === 'venture-backed startup') {
    //     return ['global market', 'multi-region focus'].at(Math.floor(Math.random() * 2))
    //   }
    //   return ['single region focus', 'local market'].at(Math.floor(Math.random() * 2))
    // })()

    const prompt =`You are tasked with creating an innovative business idea that can quickly spun up by systematically analyzing various inputs and identifying opportunities. Follow these steps carefully:

    1. Review the following inputs:    
    <business_vertical>
    ${industry.industry} (part of ${industry.group} sector)
    </business_vertical>

    <product_hunt_inspiration>
    Here are some recent successful tech products for inspiration:
    ${productHuntInspirationText}
    </product_hunt_inspiration>
    
    2. Examine the given business vertical:
       a. Consider the current state of the industry and recent product hunt entries and try to be inspired by them
       b. Identify common pain points or inefficiencies
       c. Consider how AI could potentially address these issues (with a special focus on ${specialFocus})
       
    3. Identify pain points and opportunities:
       a. List the most significant pain points in the business vertical
       b. Highlight areas where AI could provide solutions
       c. Note any gaps in the market that could be filled
    
    4. Brainstorm business ideas:
       a. Generate at least 3 potential ${targetMarket} ${techFocus} business ideas that leverage AI to address the identified pain points
       b. Briefly describe each idea and its potential impact
    
    5. Evaluate and refine the best ideas:
       a. Select the most promising idea from your brainstorming session
       b. Elaborate on how it addresses the pain points and leverages AI
       c. Consider potential challenges and how they could be overcome
    
    6. Present your final business idea:
       a. Provide a concise name for the business
       b. Describe the core concept and value proposition
       c. Highlight its unique selling points and potential impact on the industry
    
    Note: The business idea should be concise and to the point. These ideas should be pratical and have a clear first MVP product. And the business should be a fairly small scoped startup for a very small team.
    
    Present your overall analysis in <analysis></analysis> tags and final concise business idea within <business_idea><business_name></business_name><business_description></business_description></business_idea> tags, using appropriate subheadings for each section. Ensure your idea is innovative, feasible, and directly addresses the pain points identified in the given business vertical while leveraging LLMs, ML, or other generative AI. Keep your description of the idea to the point and brief.`
    
    console.log(prompt)
    const response = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-latest',
      max_tokens: 1000,
      messages: [{ role: 'user', content: prompt }],
    })
    console.log(response)
    const result = 'text' in response.content[0] 
      ? response.content[0].text 
      : ((response.content[0] as unknown) as { value: string }).value

    // Extract analysis and business ideas
    const analysisMatch = result.match(/<analysis>([^]*?)<\/analysis>/)
    const analysis = analysisMatch ? analysisMatch[1].trim() : ''

    const businessIdeas = result.match(/<business_idea>([^]*?)<\/business_idea>/g)?.map(idea => {
      const nameMatch = idea.match(/<business_name>([^]*?)<\/business_name>/)
      const descMatch = idea.match(/<business_description>([^]*?)<\/business_description>/)
      return {
        name: nameMatch?.[1].trim() || '',
        description: descMatch?.[1].trim() || ''
      }
    }) || []

    // Store each business idea in the database
    for (const idea of businessIdeas) {
      await db.insert(UserGeneratedBusinessIdeas).values({
        id: uuidv4(),
        industryGroup: industry.group,
        industry: industry.industry,
        prompt: message,
        analysis,
        idea: JSON.stringify(idea),
        name: idea.name,
        description: idea.description,
      })
    }

    return NextResponse.json(businessIdeas)
  } catch (error) {
    console.error('Error generating idea:', error)
    return NextResponse.json(
      { error: 'Failed to generate idea' },
      { status: 500 }
    )
  }
} 