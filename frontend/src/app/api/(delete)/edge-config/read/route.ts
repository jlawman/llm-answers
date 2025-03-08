import { NextResponse } from 'next/server';
import { get } from '@vercel/edge-config';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const key = searchParams.get('key');

  if (!key) {
    return NextResponse.json({ error: 'Key is required' }, { status: 400 });
  }

  try {
    const value = await get(key);
    return NextResponse.json({ value });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to read value' }, { status: 500 });
  }
}