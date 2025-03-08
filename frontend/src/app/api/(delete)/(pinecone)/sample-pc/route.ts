import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function GET() {
  try {
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const indexesList = await pinecone.listIndexes();
    console.log(indexesList);

    return NextResponse.json({ success: true, indexes: indexesList });
  } catch (error) {
    console.error('Error listing Pinecone indexes:', error);
    return NextResponse.json({ success: false, error: 'Failed to list indexes' }, { status: 500 });
  }
}
