import { db } from '@/app/db'
import { BusinessIdeas } from '@/app/db/schema'
import { desc, and, gte, lte, sql } from 'drizzle-orm'
import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const page = parseInt(searchParams.get('page') || '1')
    const limit = parseInt(searchParams.get('limit') || '3')
    const startDate = searchParams.get('startDate')
    const endDate = searchParams.get('endDate')

    const offset = (page - 1) * limit

    // Build where conditions
    const whereConditions = []
    if (startDate) {
      whereConditions.push(gte(BusinessIdeas.createTs, new Date(startDate)))
    }
    if (endDate) {
      whereConditions.push(lte(BusinessIdeas.createTs, new Date(endDate)))
    }

    // Get total count for pagination
    const totalCount = await db
      .select({ count: sql<number>`count(*)` })
      .from(BusinessIdeas)
      .where(and(...whereConditions))
      .then(result => Number(result[0].count))

    // Get paginated results
    const ideas = await db
      .select()
      .from(BusinessIdeas)
      .where(and(...whereConditions))
      .orderBy(desc(BusinessIdeas.createTs))
      .limit(limit)
      .offset(offset)

    return NextResponse.json({
      ideas,
      total: totalCount,
      page,
      totalPages: Math.ceil(totalCount / limit)
    })
  } catch (error) {
    console.error('Error fetching ideas:', error)
    return NextResponse.json(
      { error: 'Failed to fetch ideas' },
      { status: 500 }
    )
  }
} 