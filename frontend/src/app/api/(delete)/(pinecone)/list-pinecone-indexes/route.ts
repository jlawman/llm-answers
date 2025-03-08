import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function GET() {
  try {
    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    console.log('Attempting to list Pinecone indexes...');
    const index_response = await pc.listIndexes();

    console.log('Pinecones listIndexes raw response:', JSON.stringify(index_response, null, 2));

    if (!index_response.indexes || index_response.indexes.length === 0) {
      console.warn('No indexes found in the Pinecone response');
    }

    const indexNames = index_response.indexes?.map(index => index.name) || [];
    console.log('Extracted index names:', indexNames);

    // Set cache control headers to prevent caching
    const headers = new Headers();
    headers.set('Cache-Control', 'no-store, max-age=0');

    return NextResponse.json({
      success: true,
      indexes: indexNames
    }, { headers });
  } catch (error) {
    console.error('Error listing Pinecone indexes:', error);
    if (error instanceof Error) {
      console.error('Error message:', error.message);
      console.error('Error stack:', error.stack);
    }
    return NextResponse.json({ error: 'Internal server error', details: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}
