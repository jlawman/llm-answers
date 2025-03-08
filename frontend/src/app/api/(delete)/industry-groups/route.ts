import { db } from '@/app/db'
import { IndustryGroups } from '@/app/db/schema'
import { NextResponse } from 'next/server'

export async function GET() {
  try {
    const groups = await db.select().from(IndustryGroups)
    return NextResponse.json(groups)
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch industry groups' }, { status: 500 })
  }
} 