import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';

export async function POST(req: Request) {
  try {
    const { indexName, namespace, recordIds } = await req.json();

    if (!indexName || !namespace || !recordIds) {
      return NextResponse.json({ error: 'Index name, namespace, and record IDs are required' }, { status: 400 });
    }

    const pc = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const index = pc.index(indexName);
    const ns = index.namespace(namespace);

    if (Array.isArray(recordIds)) {
      await ns.deleteMany(recordIds);
      return NextResponse.json({
        success: true,
        message: `${recordIds.length} records deleted from namespace '${namespace}' in index '${indexName}'`
      });
    } else {
      await ns.deleteOne(recordIds);
      return NextResponse.json({
        success: true,
        message: `Record '${recordIds}' deleted from namespace '${namespace}' in index '${indexName}'`
      });
    }
  } catch (error) {
    console.error('Error deleting records from Pinecone index:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
