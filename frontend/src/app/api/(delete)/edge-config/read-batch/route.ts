import { NextResponse } from 'next/server';
import { getAll } from '@vercel/edge-config';

export async function POST(request: Request) {
  try {
    const { keys } = await request.json();

    if (!Array.isArray(keys) || keys.length === 0) {
      return NextResponse.json({ error: 'Invalid or empty keys array' }, { status: 400 });
    }

    const batchValues = await getAll(keys);
    return NextResponse.json(batchValues);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to read batch values' }, { status: 500 });
  }
}
