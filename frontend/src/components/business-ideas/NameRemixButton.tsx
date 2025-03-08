import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Loader2, Dices } from 'lucide-react'
import { toast } from 'sonner'

interface NameRemixButtonProps {
  originalNames: string[]
  onNewNames: (names: string[], animate?: boolean) => void
  brandSurvey?: string
}

export function NameRemixButton({ originalNames, onNewNames, brandSurvey }: NameRemixButtonProps) {
  const [isRemixing, setIsRemixing] = useState(false)

  const handleRemix = async () => {
    setIsRemixing(true)
    try {
      const prompt = `You are a creative brand naming consultant. I have some existing brand name suggestions that I'd like to remix into new variations.

${brandSurvey ? `<brand_survey>\n${brandSurvey}\n</brand_survey>` : ''}

Here are the original names:

<original_names>
${originalNames.join('\n')}
</original_names>

Please generate new brand name variations that maintain a similar style and theme but offer fresh alternatives. Each name should be domain-friendly and suitable for a SaaS business.

Keep the same organization and categories of the original names. But don't use any of the original names in your response (even if you change the domain suffix).

Your overall response should be in <result> tags.`

      const response = await fetch('/api/ask-anthropic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: prompt,
          model: 'claude-3-5-sonnet-latest'
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        console.error('API Error:', errorData)
        throw new Error(`Failed to remix names: ${response.status}`)
      }

      const data = await response.json()
      
      // Extract names from the response
      const result = data.result || ''  // Get the result string from the response
      
      if (typeof result !== 'string') {
        throw new Error(`Unexpected response format. Got ${typeof result} instead of string`)
      }

      // Parse the response to extract names, keeping section headers
      const newNames = result
        .split('\n')
        .filter(Boolean)
        .map(line => line.trim())
        .filter(line => line.length > 0)


      if (newNames.length === 0) {
        throw new Error('No names generated')
      }

      onNewNames([result], true)
    } catch (error) {
      console.error('Error remixing names:', error)
      toast.error('Failed to remix names')
    } finally {
      setIsRemixing(false)
    }
  }

  return (
    <Button
      size="sm"
      variant="outline"
      onClick={handleRemix}
      disabled={isRemixing}
    >
      {isRemixing ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <Dices className="h-4 w-4" />
      )}
    </Button>
  )
} 