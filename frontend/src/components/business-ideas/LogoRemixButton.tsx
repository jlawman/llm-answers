import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Loader2, Dices, Check, Copy } from 'lucide-react'
import {
  Dialog,
  DialogTitle,
  DialogBody,
  DialogActions,
} from "@/components/ui/dialog"
import ReactMarkdown from 'react-markdown'
import { LogoGenerationModal } from './LogoGenerationModal'

interface LogoRemixOptions {
  style: 'Simpler' | 'More Detailed' | 'Bolder' | 'More Minimal' | 'No Change'
  color: 'Warmer' | 'Cooler' | 'Monochrome' | 'Vibrant' | 'Pastel' | 'No Change'
  mood: 'Professional' | 'Playful' | 'Modern' | 'Classic' | 'Edgy' | 'No Change'
  element: 'Remove Text' | 'Geometric' | 'Organic' | 'No Change'
  composition: 'Centered' | 'Asymmetric' | 'Horizontal' | 'Vertical' | 'Circular' | 'No Change'
}

interface LogoRemixButtonProps {
  originalDescription: string
  onRemix?: (newDescription: string) => void
}

export function LogoRemixButton({ originalDescription, onRemix }: LogoRemixButtonProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [isRemixing, setIsRemixing] = useState(false)
  const [remixedDescription, setRemixedDescription] = useState<string | null>(null)
  const [copiedToClipboard, setCopiedToClipboard] = useState(false)
  const [remixOptions, setRemixOptions] = useState<LogoRemixOptions>({
    style: 'Bolder',
    color: 'Vibrant',
    mood: 'Modern',
    element: 'No Change',
    composition: 'No Change'
  })

  const handleRemix = async () => {
    setIsRemixing(true)
    try {
      const prompt = `You are tasked with rewriting an existing description for an image generator of a logo for a business. You will be given the original description and five inputs for changes: style, color, mood, element, and composition. Your goal is to create a new description that incorporates these changes while maintaining the essence of the original logo concept.

Here is the original description:
<original_description>
${originalDescription}
</original_description>

To rewrite the description, follow these steps:

1. Style Change:
Analyze the ${remixOptions.style} input. This represents a new style direction for the logo. Incorporate this style into the description by adjusting relevant parts of the original text. For example, if the style change is "minimalist", you might remove complex details and focus on simple, clean lines.

2. Color Change:
Review the ${remixOptions.color} input. This indicates a desired change in the color scheme. Modify the color-related aspects of the description to reflect this change. If specific colors are mentioned in the original description, replace them with new colors that align with this direction.

3. Mood Change:
Consider the ${remixOptions.mood} input. This sets the overall tone and feel of the logo. Adjust the description to reflect this mood while maintaining brand appropriateness.

4. Element Change:
Apply the ${remixOptions.element} input. This suggests an addition, removal, or modification of a specific element in the logo. Incorporate this change by either introducing a new element, removing an existing one, or altering a current element as specified.

5. Composition Change:
Implement the ${remixOptions.composition} input. This determines how the elements should be arranged in relation to each other. Adjust the layout and positioning description accordingly.

6. Maintain Coherence:
Ensure that the new description remains coherent and logical. The changes should blend seamlessly with the unchanged parts of the original description. If any part of the original description conflicts with the new changes, modify it to maintain consistency.

7. Preserve Key Concepts:
While making changes, be sure to preserve the core concept and key identifying features of the original logo description, unless the changes explicitly override them.

8. Clarity and Conciseness:
Keep the new description clear and concise. Avoid unnecessary elaboration or repetition. The description should provide a clear mental image of the modified logo.

9. Output Format:
Present your rewritten description within <new_description> tags. After the description, provide a brief explanation of the changes you made and how they align with the requested modifications within <explanation> tags.

10. Remember the tool:
The tool is an image generator. It will generate images based on your descriptions and cannot consistently generate text so do not use text or wordmarks in your descriptions.

Remember, your goal is to create a new description that incorporates the requested changes while maintaining the essence of the original logo concept. Be creative in your rewriting, but ensure that the final description is practical for an image generator to interpret.`

      const response = await fetch('/api/ask-anthropic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt })
      })

      if (!response.ok) throw new Error('Failed to remix logo')

      const data = await response.json()
      const newDescription = data.new_description || data.result?.new_description
      
      if (!newDescription) {
        throw new Error('No description found in response')
      }
      
      setRemixedDescription(newDescription)
      
      if (onRemix) {
        onRemix(newDescription)
      }
    } catch (error) {
      console.error('Error remixing logo:', error)
    } finally {
      setIsRemixing(false)
    }
  }

  const handleCopy = () => {
    if (!remixedDescription) return
    
    navigator.clipboard.writeText(remixedDescription).then(() => {
      setCopiedToClipboard(true)
      setTimeout(() => setCopiedToClipboard(false), 2000)
    })
  }

  const handleClose = () => {
    setIsOpen(false)
    setRemixedDescription(null)
    setCopiedToClipboard(false)
  }

  return (
    <>
      <Button size="sm" variant="outline" className="gap-2" onClick={() => setIsOpen(true)}>
        <Dices className="h-4 w-4" />
        Remix
      </Button>

      <Dialog 
        open={isOpen} 
        onClose={handleClose}
        className="w-[900px] max-w-[90vw]"
      >
        <DialogTitle>
          Remix Logo Options
        </DialogTitle>
        
        <DialogBody>
          <div className="grid gap-6">
            {remixedDescription ? (
              <div className="grid grid-cols-2 gap-6">
                <div className="prose prose-sm max-w-none">
                  <h3 className="text-sm font-medium mb-2">Original Design</h3>
                  <div className="p-4 rounded-md bg-gray-50 min-h-[200px] overflow-y-auto">
                    <ReactMarkdown>{originalDescription}</ReactMarkdown>
                  </div>
                </div>
                
                <div className="prose prose-sm max-w-none">
                  <h3 className="text-sm font-medium mb-2">Remixed Design</h3>
                  <div className="p-4 rounded-md bg-gray-50 min-h-[200px] overflow-y-auto">
                    <ReactMarkdown>{remixedDescription}</ReactMarkdown>
                    <div className="mt-4 flex gap-2">
                      <LogoGenerationModal prompt={remixedDescription} />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={handleCopy}
                      >
                        {copiedToClipboard ? (
                          <Check className="h-4 w-4" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid gap-4">
                <div className="grid gap-2">
                  <label className="text-sm font-medium">Style Adjustment</label>
                  <select
                    className="w-full p-2 border rounded-md"
                    value={remixOptions.style}
                    onChange={(e) => setRemixOptions(prev => ({
                      ...prev,
                      style: e.target.value as LogoRemixOptions['style']
                    }))}
                  >
                    <option value="Simpler">Simpler</option>
                    <option value="More Detailed">More Detailed</option>
                    <option value="Bolder">Bolder</option>
                    <option value="More Minimal">More Minimal</option>
                    <option value="No Change">No Change</option>
                  </select>
                </div>

                <div className="grid gap-2">
                  <label className="text-sm font-medium">Color Direction</label>
                  <select
                    className="w-full p-2 border rounded-md"
                    value={remixOptions.color}
                    onChange={(e) => setRemixOptions(prev => ({
                      ...prev,
                      color: e.target.value as LogoRemixOptions['color']
                    }))}
                  >
                    <option value="Warmer">Warmer</option>
                    <option value="Cooler">Cooler</option>
                    <option value="Monochrome">Monochrome</option>
                    <option value="Vibrant">Vibrant</option>
                    <option value="Pastel">Pastel</option>
                    <option value="No Change">No Change</option>
                  </select>
                </div>

                <div className="grid gap-2">
                  <label className="text-sm font-medium">Mood</label>
                  <select
                    className="w-full p-2 border rounded-md"
                    value={remixOptions.mood}
                    onChange={(e) => setRemixOptions(prev => ({
                      ...prev,
                      mood: e.target.value as LogoRemixOptions['mood']
                    }))}
                  >
                    <option value="Professional">Professional</option>
                    <option value="Playful">Playful</option>
                    <option value="Modern">Modern</option>
                    <option value="Classic">Classic</option>
                    <option value="No Change">No Change</option>
                  </select>
                </div>

                <div className="grid gap-2">
                  <label className="text-sm font-medium">Element Changes</label>
                  <select
                    className="w-full p-2 border rounded-md"
                    value={remixOptions.element}
                    onChange={(e) => setRemixOptions(prev => ({
                      ...prev,
                      element: e.target.value as LogoRemixOptions['element']
                    }))}
                  >
                    <option value="Remove Text">Remove Text</option>
                    <option value="Geometric">Geometric</option>
                    <option value="Organic">Organic</option>
                    <option value="No Change">No Change</option>
                  </select>
                </div>

                <div className="grid gap-2">
                  <label className="text-sm font-medium">Composition</label>
                  <select
                    className="w-full p-2 border rounded-md"
                    value={remixOptions.composition}
                    onChange={(e) => setRemixOptions(prev => ({
                      ...prev,
                      composition: e.target.value as LogoRemixOptions['composition']
                    }))}
                  >
                    <option value="Centered">Centered</option>
                    <option value="Asymmetric">Asymmetric</option>
                    <option value="Horizontal">Horizontal</option>
                    <option value="Vertical">Vertical</option>
                    <option value="Circular">Circular</option>
                    <option value="No Change">No Change</option>
                  </select>
                </div>
              </div>
            )}
          </div>
        </DialogBody>

        <DialogActions>
          <Button 
            variant="outline" 
            onClick={handleClose}
          >
            Close
          </Button>
          
          {!remixedDescription && (
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
                'Remix This Logo'
              )}
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </>
  )
} 