import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { indexName } = await req.json();

    if (!indexName) {
      return NextResponse.json({ error: 'Index name is required' }, { status: 400 });
    }

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const index = pc.index(indexName);
    const stats = await index.describeIndexStats();

    console.log(stats);

    return NextResponse.json({
      success: true,
      stats: {
        namespaces: stats.namespaces,
        dimension: stats.dimension,
        indexFullness: stats.indexFullness,
        totalRecordCount: stats.totalRecordCount
      }
    });
  } catch (error) {
    console.error('Error describing Pinecone index:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
