import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { indexName, namespace } = await req.json();

    if (!indexName || !namespace) {
      return NextResponse.json({ error: 'Index name and namespace are required' }, { status: 400 });
    }

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const index = pc.index(indexName);
    await index.namespace(namespace).deleteAll();

    return NextResponse.json({
      success: true,
      message: `All records deleted from namespace '${namespace}' in index '${indexName}'`
    });
  } catch (error) {
    console.error('Error deleting records from Pinecone index:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
