import { NextResponse } from 'next/server';
import { jobManager } from '@/lib/jobManager';

export async function POST() {
  const jobId = jobManager.createJob();
  jobManager.startJob(jobId);
  return NextResponse.json({ jobId });
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const jobId = searchParams.get('id');
  
  if (!jobId) {
    return NextResponse.json({ error: 'Job ID is required' }, { status: 400 });
  }

  const job = jobManager.getJobStatus(jobId);
  
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  return NextResponse.json(job);
}