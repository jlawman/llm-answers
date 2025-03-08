import { NextResponse } from 'next/server';
import { getAll } from '@vercel/edge-config';

export async function GET() {
  try {
    const allValues = await getAll();
    return NextResponse.json(allValues);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to read all values' }, { status: 500 });
  }
}
