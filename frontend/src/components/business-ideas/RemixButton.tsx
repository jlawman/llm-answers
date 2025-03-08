import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Loader2, Dices, Check, Copy } from 'lucide-react'
import {
  Dialog,
  DialogTitle,
  DialogBody,
  DialogActions,
} from "@/components/ui/dialog"
import ReactMarkdown from 'react-markdown'

interface BusinessIdea {
  name: string
  description: string
  industry: string
  industryGroup: string
  createTs: Date
}

interface RemixOptions {
  customerType: 'B2B' | 'B2C' | 'Random'
  targetIndustry: 'Similar' | 'Different'
  scale: 'Similar' | 'Simpler' | 'More Expansive'
  marketRegion: 'Local' | 'National' | 'Global'
  additionalContext?: string
}

interface RemixButtonProps {
  idea: BusinessIdea
  onNewIdea: (idea: BusinessIdea | null) => void
}

interface RemixedIdeaContent {
  businessIdea: string
  relation: string
}

export function RemixButton({ idea, onNewIdea }: RemixButtonProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isRemixing, setIsRemixing] = useState(false)
  const [remixedIdea, setRemixedIdea] = useState<BusinessIdea | null>(null)
  const [copiedToClipboard, setCopiedToClipboard] = useState(false)
  const [industryGroups, setIndustryGroups] = useState<{ industryGroup: string, industries: string }[]>([])
  const [remixOptions, setRemixOptions] = useState<RemixOptions>({
    customerType: 'Random',
    targetIndustry: 'Different',
    scale: 'Similar',
    marketRegion: 'National',
  })
  const [remixedContent, setRemixedContent] = useState<RemixedIdeaContent | null>(null)

  useEffect(() => {
    const fetchIndustryGroups = async () => {
      const response = await fetch('/api/industry-groups')
      const data = await response.json()
      setIndustryGroups(data)
    }
    fetchIndustryGroups()
  }, [])

  const getRandomIndustry = () => {
    if (industryGroups.length === 0) return { industry: idea.industry, industryGroup: idea.industryGroup }
    
    if (remixOptions.targetIndustry === 'Similar') {
      // Find the current industry group
      const currentGroup = industryGroups.find(g => g.industryGroup === idea.industryGroup)
      if (currentGroup) {
        const industries = currentGroup.industries.split(',').map(i => i.trim())
        // Filter out the current industry and select a random one from the same group
        const otherIndustries = industries.filter(i => i !== idea.industry)
        if (otherIndustries.length > 0) {
          const randomIndustry = otherIndustries[Math.floor(Math.random() * otherIndustries.length)]
          return {
            industry: randomIndustry,
            industryGroup: currentGroup.industryGroup
          }
        }
      }
    }
    
    // For 'Different' option or fallback if same group selection fails
    const randomGroup = industryGroups[Math.floor(Math.random() * industryGroups.length)]
    const industries = randomGroup.industries.split(',').map(i => i.trim())
    const randomIndustry = industries[Math.floor(Math.random() * industries.length)]
    
    return {
      industry: randomIndustry,
      industryGroup: randomGroup.industryGroup
    }
  }

  const handleRemix = async () => {
    setIsRemixing(true)
    try {
      const targetIndustry = remixOptions.targetIndustry === 'Different' 
        ? getRandomIndustry()
        : { industry: idea.industry, industryGroup: idea.industryGroup }

      const prompt = `You are a creative business consultant tasked with generating new business ideas based on existing concepts. Your goal is to come up with an innovative business idea that remixes an existing business while adhering to specific parameters.

First, carefully read and analyze the existing business idea:

<existing_business_idea>
Original idea: "${idea.name}" in the ${idea.industry} industry.
Original description: ${idea.description}
</existing_business_idea>

Now, consider the following remix options that will guide your ideation process:

<remix_options>
- Business Model: ${remixOptions.customerType}
- Industry Approach: ${remixOptions.targetIndustry === 'Similar' ? `Stay close to ${idea.industry}` : 'Explore a different industry'}
- Scale/Scope: ${remixOptions.scale} to the original
- Market Region: ${remixOptions.marketRegion}
</remix_options>

<user_context>
${remixOptions.additionalContext ? `- Additional Context: ${remixOptions.additionalContext}` : ''}
</user_context>

These options define the parameters for your new business idea:
- customerType: Determines whether the new business should target other businesses (B2B), consumers (B2C), or a random choice between the two.
- targetIndustry: Indicates whether the new business should operate in a similar industry or a different one.
- scale: Suggests whether the new business should be of similar scale, simpler, or more expansive compared to the original idea.
- marketRegion: Specifies the geographical focus of the new business (local, national, or global).
- additionalContext: Provides any extra information or constraints to consider (if applicable).

Analyze the existing business idea, paying attention to its core concepts, target market, and unique selling propositions. Then, use your creativity and business acumen to generate a new business idea that incorporates elements of the original concept while adhering to the specified remix options.

In your response, provide the following:

1. A brief description of the new business idea (2-3 sentences).
2. An explanation of how the new idea relates to the original business concept.

Present your response in the following format:

<result>
<business_name>
[Name of the new business idea]
</business_name>
<new_business_idea>
[Brief description of the new business idea with any markdown formatting for clarity]
</new_business_idea>

<relation_to_original>
[Extremely brief Explanation of how the new idea relates to the original business concept]
</relation_to_original>

<industry>
${targetIndustry.industry}
</industry>
<industry_group>
${targetIndustry.industryGroup}
</industry_group>

</result>

Be creative, innovative, and ensure that your new business idea is both practical and aligned with the given remix options.`


      const response = await fetch('/api/ask-anthropic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt })
      })

      if (!response.ok) throw new Error('Failed to remix idea')

      const data = await response.json()
      
      // Extract the business idea, relation text, and industry info from the response
      const businessNameMatch = data.result.match(/<business_name>[\s\n]*([^<]+?)[\s\n]*<\/business_name>/)
      const businessIdeaMatch = data.result.match(/<new_business_idea>([\s\S]*?)<\/new_business_idea>/)
      const relationMatch = data.result.match(/<relation_to_original>([\s\S]*?)<\/relation_to_original>/)
      const industryMatch = data.result.match(/<industry>([\s\S]*?)<\/industry>/)
      const industryGroupMatch = data.result.match(/<industry_group>([\s\S]*?)<\/industry_group>/)
      
      const businessName = businessNameMatch ? businessNameMatch[1].trim() : "New Remixed Idea"
      const businessIdeaText = businessIdeaMatch ? businessIdeaMatch[1].trim() : ''
      const relationText = relationMatch ? relationMatch[1].trim() : ''
      const newIndustry = industryMatch ? industryMatch[1].trim() : targetIndustry.industry
      const newIndustryGroup = industryGroupMatch ? industryGroupMatch[1].trim() : targetIndustry.industryGroup

      // Store both parts of the content
      setRemixedContent({
        businessIdea: businessIdeaText,
        relation: relationText
      })

      // Create new idea with extracted industry information and business name
      const newIdea: BusinessIdea = {
        name: businessName,
        description: businessIdeaText,
        industry: newIndustry,
        industryGroup: newIndustryGroup,
        createTs: new Date()
      }

      setRemixedIdea(newIdea)
      onNewIdea(newIdea)
    } catch (error) {
      console.error('Error remixing idea:', error)
    } finally {
      setIsRemixing(false)
    }
  }

  const handleCopy = () => {
    if (!remixedIdea) return
    
    const text = `${remixedIdea.name}\n\n${remixedIdea.description}`
    navigator.clipboard.writeText(text).then(() => {
      setCopiedToClipboard(true)
      setTimeout(() => setCopiedToClipboard(false), 2000)
    })
  }

  const handleClose = () => {
    setIsOpen(false)
    setRemixedIdea(null)
    setCopiedToClipboard(false)
  }

  return (
    <>
      <Button size="sm" variant="outline" className="gap-2" onClick={() => setIsOpen(true)}>
        <Dices className="h-4 w-4" />
        Remix
      </Button>

      <Dialog open={isOpen} onClose={handleClose}>
        <DialogTitle>
          {remixedIdea ? 'Remixed Business Idea' : 'Remix Business Idea'}
        </DialogTitle>
        
        <DialogBody>
          {remixedIdea ? (
            <div className="space-y-4">
              <div className="prose prose-sm max-w-none">
                <h3 className="text-xl font-bold mb-4">{remixedIdea.name}</h3>
                <ReactMarkdown>
                  {remixedContent?.businessIdea || ''}
                </ReactMarkdown>
                
                <h3 className="text-xl font-bold mt-6 mb-4">Relation to Original</h3>
                <ReactMarkdown>
                  {remixedContent?.relation || ''}
                </ReactMarkdown>
              </div>
              
              <div className="text-sm text-gray-500 flex gap-4">
                <span className="flex items-center gap-1">
                  <span className="font-medium">Industry:</span> {remixedIdea.industry}
                </span>
                <span className="flex items-center gap-1">
                  <span className="font-medium">Group:</span> {remixedIdea.industryGroup}
                </span>
              </div>
            </div>
          ) : (
            <div className="grid gap-4">
              <div className="grid gap-2">
                <label className="text-sm font-medium">Business Model</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={remixOptions.customerType}
                  onChange={(e) => setRemixOptions(prev => ({
                    ...prev,
                    customerType: e.target.value as RemixOptions['customerType']
                  }))}
                >
                  <option value="B2B-or-B2C">Random</option>
                  <option value="B2B">B2B</option>
                  <option value="B2C">B2C</option>
                </select>
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Industry</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={remixOptions.targetIndustry}
                  onChange={(e) => setRemixOptions(prev => ({
                    ...prev,
                    targetIndustry: e.target.value as RemixOptions['targetIndustry']
                  }))}
                >
                  <option value="Similar">Similar</option>
                  <option value="Different">Something Different</option>
                </select>
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Scale/Scope</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={remixOptions.scale}
                  onChange={(e) => setRemixOptions(prev => ({
                    ...prev,
                    scale: e.target.value as RemixOptions['scale']
                  }))}
                >
                  <option value="Similar">Similar</option>
                  <option value="Simpler">Simpler</option>
                  <option value="More Expansive">More Expansive</option>
                </select>
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Market Region</label>
                <select
                  className="w-full p-2 border rounded-md"
                  value={remixOptions.marketRegion}
                  onChange={(e) => setRemixOptions(prev => ({
                    ...prev,
                    marketRegion: e.target.value as RemixOptions['marketRegion']
                  }))}
                >
                  <option value="Local">Local</option>
                  <option value="National">National</option>
                  <option value="Global">Global</option>
                </select>
              </div>

              <div className="grid gap-2">
                <label className="text-sm font-medium">Additional Context (Optional)</label>
                <Input
                  placeholder="Any other considerations..."
                  value={remixOptions.additionalContext}
                  onChange={(e) => setRemixOptions(prev => ({
                    ...prev,
                    additionalContext: e.target.value
                  }))}
                />
              </div>
            </div>
          )}
        </DialogBody>

        <DialogActions>
          <Button 
            variant="outline" 
            onClick={handleClose}
          >
            {remixedIdea ? 'Close' : 'Cancel'}
          </Button>
          
          {remixedIdea ? (
            <>
              <Button
                variant="outline"
                onClick={handleCopy}
                className="gap-2"
              >
                {copiedToClipboard ? (
                  <>
                    <Check className="h-4 w-4" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    
                  </>
                )}
              </Button>
              <Button onClick={() => {
                setRemixedIdea(null)
                handleRemix()
              }}>
                Remix Again
              </Button>
            </>
          ) : (
            <Button 
              onClick={handleRemix}
              disabled={isRemixing}
            >
              {isRemixing ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Remixing...
                </>
              ) : (
                'Remix This Idea'
              )}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </>
  )
} 