import { NextResponse } from 'next/server';
import { has } from '@vercel/edge-config';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const key = searchParams.get('key');

  if (!key) {
    return NextResponse.json({ error: 'Key is required' }, { status: 400 });
  }

  try {
    const exists = await has(key);
    return NextResponse.json({ exists });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to check if key exists' }, { status: 500 });
  }
}