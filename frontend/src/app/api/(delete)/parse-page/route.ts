import { NextResponse } from 'next/server'

export async function POST(request: Request) {
    const { url } = await request.json()

    const requestBody = {
        "url": url,
        "proxy_location": null,
        "async": false, 
        "max_cache_age": 0,
        "settings": {
            "record_request": false,
            "actions": [
                {
                    "type": "generate_markdown"
                }
            ]
        }
    }

    const response = await fetch('https://api.gaffa.dev/v1/browser/requests', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': process.env.GAFFA_API_KEY!
        },
        body: JSON.stringify(requestBody)
    })

    const data = await response.json()
    console.log(data)
    const markdownUrl = data.data.actions[0].output
    
    // Fetch the markdown content
    const markdownResponse = await fetch(markdownUrl)
    const markdownContent = await markdownResponse.text()
    
    return NextResponse.json({ markdown: markdownContent })
}