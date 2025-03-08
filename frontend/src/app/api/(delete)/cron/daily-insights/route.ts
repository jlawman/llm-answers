import { NextResponse } from 'next/server'
import Parser from 'rss-parser'
import { db } from '@/app/db'
import { ProductHuntEntries, AINews, IndustryGroups, BusinessIdeas } from '@/app/db/schema'
import { v4 as uuidv4 } from 'uuid'
import { Anthropic } from '@anthropic-ai/sdk'

const parser = new Parser()

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
})

interface ProductHuntEntry {
  id: string
  name: string
  description: string
  url: string
  publishDate: string
  author: string
  upvotes: string
  createTs: Date
}

interface AINewsEntry {
  id: string
  title: string
  published: string
  summary: string
  createTs: Date
}

interface IndustryInfo {
  group: string
  industry: string
}

export const dynamic = 'force-dynamic'
export const revalidate = 0

async function getProductHuntFeed() {
  try {
    const feed = await parser.parseURL('https://www.producthunt.com/feed')
    return feed.items.slice(0, 10).map(entry => ({
      id: uuidv4(),
      name: entry.title || '',
      description: entry.content?.replace(/<[^>]*>/g, '') || '', // Strip HTML tags
      url: entry.link || '',
      publishDate: entry.pubDate || '',
      author: entry.author || '',
      upvotes: entry.categories?.[0] || '0',
      createTs: new Date(),
    }))
  } catch (error) {
    console.error('Error fetching Product Hunt feed:', error)
    return []
  }
}

async function getLatestAINews() {
  try {
    const feed = await parser.parseURL('https://buttondown.com/ainews/rss')
    const latestEntry = feed.items[0]
    
    if (latestEntry) {
      return {
        id: uuidv4(),
        title: latestEntry.title || '',
        published: latestEntry.pubDate || '',
        summary: latestEntry.contentSnippet || '',
        createTs: new Date(),
      }
    }
    return null
  } catch (error) {
    console.error('Error fetching AI News:', error)
    return null
  }
}

async function getRandomIndustry() {
  try {
    // Get all industry groups
    const industryGroups = await db.select().from(IndustryGroups)
    
    // Select random industry group
    const randomGroup = industryGroups[Math.floor(Math.random() * industryGroups.length)]
    
    // Get random industry from the comma-separated list
    const industries = randomGroup.industries.split(',').map(i => i.trim())
    const randomIndustry = industries[Math.floor(Math.random() * industries.length)]
    
    return {
      group: randomGroup.industryGroup,
      industry: randomIndustry
    }
  } catch (error) {
    console.error('Error getting random industry:', error)
    return null
  }
}

async function generateBusinessIdea(
  products: ProductHuntEntry[],
  aiNews: AINewsEntry,
  industry: IndustryInfo
) {
  const prompt =`You are tasked with creating an innovative business idea that can quickly spun up by systematically analyzing various inputs and identifying opportunities. Follow these steps carefully:

1. Review the following inputs:

<recent_ai_news>
${aiNews.title}
${aiNews.summary}
</recent_ai_news>

<business_vertical>
${industry.industry} (part of ${industry.group} sector)
</business_vertical>

<product_hunt_inspiration>
${products.map(p => `- ${p.name}: ${p.description}`).join('\n')}
</product_hunt_inspiration>

2. Analyze the recent AI news:
   a. Identify key trends and advancements in AI
   b. Consider potential applications of these AI developments
   c. Note any challenges or limitations mentioned

3. Examine the given business vertical:
   a. Consider the current state of the industry
   b. Identify common pain points or inefficiencies
   c. Consider how AI could potentially address these issues

4. Study the Product Hunt inspiration:
   a. Analyze the product's key features and value proposition
   b. Identify what makes it unique or successful
   c. Consider how elements of this product could be applied or improved in the given business vertical

5. Identify pain points and opportunities:
   a. List the most significant pain points in the business vertical
   b. Highlight areas where AI could provide solutions
   c. Note any gaps in the market that could be filled

6. Brainstorm business ideas:
   a. Generate at least 3 potential SAAS business ideas that leverage generative AI (or traditional ML) to address the identified pain points
   b. Briefly describe each idea and its potential impact

7. Evaluate and refine the best idea:
   a. Select the most promising idea from your brainstorming session
   b. Elaborate on how it addresses the pain points and leverages AI
   c. Consider potential challenges and how they could be overcome

8. Present your final business idea:
   a. Provide a concise name for the business
   b. Describe the core concept and value proposition
   c. Explain how it incorporates AI and addresses market needs
   d. Highlight its unique selling points and potential impact on the industry

Note: The business idea should be concise and to the point. It is essential that the business idea has a clear first MVP product. And the business should be a fairly small scoped startup for a very small team that could be quickly built with coding tools.

Present your overall analysis in <analysis></analysis> tags and final business idea within <business_idea><business_name></business_name><business_description></business_description></business_idea> tags. Ensure your idea is innovative, feasible, and directly addresses the pain points identified in the given business vertical while leveraging recent generative AI.`

  try {
    const response = await anthropic.messages.create({
      model: 'claude-3-5-sonnet-latest',
      max_tokens: 1000,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    // Safely handle the content block type
    const contentBlock = response.content[0];
    const result = contentBlock.type === 'text' 
      ? contentBlock.text 
      : '';
    
    // Extract analysis
    const analysisMatch = result.match(/<analysis>([^]*?)<\/analysis>/);
    const analysis = analysisMatch ? analysisMatch[1].trim() : '';
    
    // Extract business ideas
    const businessIdeaMatches = result.matchAll(/<business_idea>([^]*?)<\/business_idea>/g);
    const businessIdeas = Array.from(businessIdeaMatches).map(match => {
      const ideaText = match[1].trim();
      
      // Extract name and description using their specific tags
      const nameMatch = ideaText.match(/<business_name>([^]*?)<\/business_name>/);
      const descriptionMatch = ideaText.match(/<business_description>([^]*?)<\/business_description>/);
      
      return {
        id: uuidv4(),
        industryGroup: industry.group,
        industry: industry.industry,
        prompt: prompt,
        analysis: analysis,
        idea: ideaText,
        name: nameMatch ? nameMatch[1].trim() : 'Unnamed Idea',
        description: descriptionMatch ? descriptionMatch[1].trim() : 'No description provided',
        createTs: new Date(),
      };
    });
    
    // Save all business ideas to the database
    if (businessIdeas.length > 0) {
      await db.insert(BusinessIdeas).values(businessIdeas);
    }
    
    return { 
      analysis,
      ideas: businessIdeas
    };
  } catch (error) {
    console.error('Error generating business idea:', error);
    return null;
  }
}

export async function GET() {
  try {
    // Fetch all data
    const [products, aiNews, randomIndustry] = await Promise.all([
      getProductHuntFeed(),
      getLatestAINews(),
      getRandomIndustry()
    ])

    // Store in database
    if (products.length > 0) {
      await db.insert(ProductHuntEntries).values(products)
    }

    if (aiNews) {
      await db.insert(AINews).values(aiNews)
    }

    // Generate business idea if we have all required data
    let businessIdea = null
    if (products.length > 0 && aiNews && randomIndustry) {
      businessIdea = await generateBusinessIdea(products, aiNews, randomIndustry)
    }

    return NextResponse.json({ 
      success: true, 
      message: 'Daily insights stored successfully',
      businessIdea
    })
  } catch (error) {
    console.error('Error in daily insights cron:', error)
    return NextResponse.json(
      { success: false, error: 'Failed to store daily insights' },
      { status: 500 }
    )
  }
} 