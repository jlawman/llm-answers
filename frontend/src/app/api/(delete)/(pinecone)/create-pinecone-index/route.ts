import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { indexName = 'quickstart', dimension = 1024, metric = 'cosine' } = await req.json();

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    await pc.createIndex({
      name: indexName,
      dimension: dimension,
      metric: metric,
      spec: {
        serverless: {
          cloud: 'aws',
          region: 'us-east-1'
        }
      }
    });

    return NextResponse.json({ success: true, message: 'Index created successfully' });
  } catch (error) {
    console.error('Error creating Pinecone index:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
