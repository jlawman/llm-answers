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

    await pc.deleteIndex(indexName);

    return NextResponse.json({ success: true, message: 'Index deleted successfully' });
  } catch (error) {
    console.error('Error deleting Pinecone index:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
