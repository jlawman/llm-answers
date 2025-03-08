import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { indexName, data }: { indexName: string, data: { id: string, text: string }[] } = await req.json();

    if (!indexName || !data || data.length === 0) {
      return NextResponse.json({ error: 'Invalid input data' }, { status: 400 });
    }

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const model = 'multilingual-e5-large';

    const embeddings = await pc.inference.embed(
      model,
      data.map(d => d.text),
      { inputType: 'passage', truncate: 'END' }
    );

    const index = pc.index(indexName);

    const vectors = data.map((d, i) => {
      const embeddingValues = embeddings[i]?.values;
      if (!embeddingValues) {
        throw new Error(`No embedding values for index ${i}`);
      }
      return {
        id: d.id,
        values: embeddingValues,
        metadata: { text: d.text }
      };
    });

    await index.namespace('ns1').upsert(vectors);

    return NextResponse.json({ success: true, message: 'Data embedded and upserted successfully' });
  } catch (error) {
    console.error('Error embedding and upserting data to Pinecone:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
