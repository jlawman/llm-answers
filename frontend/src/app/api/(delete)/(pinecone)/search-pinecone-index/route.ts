import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { indexName, query, namespace } = await req.json();

    // Check for missing required parameters
    if (!indexName || !query || !namespace) {
      const missingParams = [];
      if (!indexName) missingParams.push('indexName');
      if (!query) missingParams.push('query');
      if (!namespace) missingParams.push('namespace');
      
      const errorMessage = `Missing required parameter(s): ${missingParams.join(', ')}`;
      return NextResponse.json({ error: errorMessage }, { status: 400 });
    }

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const model = 'multilingual-e5-large';

    const embedding = await pc.inference.embed(
      model,
      [query],
      { inputType: 'query' }
    );

    const index = pc.index(indexName);

    const queryResponse = await index.namespace(namespace).query({
      topK: 3,
      vector: embedding[0]?.values ?? [],
      includeValues: false,
      includeMetadata: true
    });

    console.log(queryResponse);

    // Extract text from metadata and include it in the results
    const results = queryResponse.matches.map(match => ({
      text: match.metadata?.text || 'No text available',
      score: match.score
    }));

    return NextResponse.json({ success: true, results });
  } catch (error) {
    console.error('Error searching Pinecone index:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
